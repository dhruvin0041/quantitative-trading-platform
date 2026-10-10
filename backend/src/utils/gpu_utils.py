"""Centralized GPU device management and benchmarking utilities."""
import logging
import os
import platform
import time
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect and return the best available compute device with logging."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        try:
            # Dynamically verify if active CUDA architecture (e.g., sm_120 on RTX 5070)
            # has compiled kernel binaries in this PyTorch installation
            test_t = torch.zeros(1, device="cuda")
            del test_t
            logger.info("PyTorch GPU active: %s | VRAM: %.2f GB", gpu_name, vram_total)
            return torch.device("cuda")
        except Exception as e:
            logger.warning(
                "GPU detected (%s, %.2f GB VRAM), but PyTorch CUDA kernel execution failed (%s). "
                "Routing PyTorch to multi-threaded CPU. Tree ensembles (XGBoost/CatBoost) will continue using GPU.",
                gpu_name,
                vram_total,
                e,
            )
            return torch.device("cpu")
    else:
        logger.warning("No CUDA GPU detected. Falling back to CPU.")
        return torch.device("cpu")



def configure_gpu_optimizations():
    """Enable global GPU performance optimizations."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(
            "GPU optimizations enabled: cuDNN benchmark=True, TF32=True"
        )
    else:
        # CPU Maximization Fallback
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
        logger.info(f"No GPU. Maximizing CPU usage: {cpu_count} threads allocated.")

def get_compute_backend() -> dict:
    """Detects available hardware and configures the optimal execution backend."""
    cpu_count = os.cpu_count() or 4
    cpu_name = platform.processor() or "Unknown CPU"

    if torch.cuda.is_available():
        backend = "CUDA"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        cuda_version = torch.version.cuda
    else:
        backend = "CPU"
        gpu_name = "None"
        vram_gb = 0.0
        cuda_version = "None"

    configure_gpu_optimizations()

    report = {
        "CPU": cpu_name,
        "Cores": cpu_count, # Hardware threads
        "Threads": cpu_count,
        "GPU": gpu_name,
        "CUDA Version": cuda_version,
        "VRAM": f"{vram_gb} GB",
        "Backend Selected": backend
    }

    print("-" * 40)
    print("COMPUTE BACKEND REPORT")
    print("-" * 40)
    for k, v in report.items():
        print(f"{k}: {v}")
    print("-" * 40)

    return report


def get_gpu_memory_info() -> dict:
    """Return current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 4),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 4),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 4),
    }


def configure_tensorflow_gpu():
    """Configure TensorFlow/Keras to use GPU with memory growth."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(
                "TensorFlow GPU configured: %d GPU(s) detected with memory growth enabled",
                len(gpus),
            )
            # Enable mixed precision for TF/Keras
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("TensorFlow mixed precision (float16) enabled")
        else:
            cpu_count = os.cpu_count() or 4
            tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
            tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
            logger.warning(f"TensorFlow: No GPU detected, using {cpu_count} CPU threads.")
    except Exception as e:
        logger.warning("TensorFlow GPU configuration failed: %s", e)


def get_xgboost_gpu_params() -> dict:
    """Return GPU-optimized params for XGBoost if CUDA is available, else max CPU."""
    if torch.cuda.is_available():
        logger.info("XGBoost: Using GPU acceleration (device=cuda, tree_method=hist)")
        return {"device": "cuda", "tree_method": "hist"}
    return {"n_jobs": os.cpu_count() or -1}


def get_catboost_gpu_params() -> dict:
    """Return GPU-optimized params for CatBoost if CUDA is available, else max CPU."""
    if torch.cuda.is_available():
        logger.info("CatBoost: Using GPU acceleration (task_type=GPU)")
        return {"task_type": "GPU", "devices": "0"}
    return {"thread_count": os.cpu_count() or -1}


_LGBM_GPU_SUPPORTED = None


def get_lightgbm_gpu_params() -> dict:
    """Return GPU-optimized params for LightGBM if GPU build is available, else max CPU."""
    global _LGBM_GPU_SUPPORTED
    if _LGBM_GPU_SUPPORTED is None:
        try:
            import lightgbm as lgb
            # Test if this build has CUDA capability compiled in
            test_clf = lgb.LGBMClassifier(device="cuda", n_estimators=1, verbose=-1)
            test_clf.fit([[0]], [0])
            _LGBM_GPU_SUPPORTED = "cuda"
        except Exception:
            try:
                import lightgbm as lgb
                test_clf = lgb.LGBMClassifier(device="gpu", n_estimators=1, verbose=-1)
                test_clf.fit([[0]], [0])
                _LGBM_GPU_SUPPORTED = "gpu"
            except Exception:
                _LGBM_GPU_SUPPORTED = False

    if _LGBM_GPU_SUPPORTED == "cuda":
        logger.info("LightGBM: Using CUDA acceleration")
        return {"device": "cuda"}
    elif _LGBM_GPU_SUPPORTED == "gpu":
        logger.info("LightGBM: Using OpenCL GPU acceleration")
        return {"device": "gpu", "gpu_use_dp": False}

    return {"n_jobs": os.cpu_count() or -1}



@contextmanager
def benchmark_context(label: str):
    """Context manager that logs wall-clock time and GPU memory for a block."""
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    if gpu_available:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    msg = f"[BENCHMARK] {label}: {elapsed:.3f}s"
    if gpu_available:
        peak_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
        msg += f" | Peak VRAM: {peak_mb:.1f} MB"
    logger.info(msg)
    print(msg)


def verify_gpu_utilization():
    """Print comprehensive GPU verification report."""
    print("=" * 60)
    print("GPU UTILIZATION VERIFICATION REPORT")
    print("=" * 60)

    # PyTorch
    print("\n[PyTorch]")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        info = get_gpu_memory_info()
        print(f"  Total VRAM: {info['total_vram_gb']} GB")
        print(f"  Allocated: {info['allocated_gb']} GB")
        print(f"  Peak: {info['peak_allocated_gb']} GB")

    # TensorFlow
    try:
        import tensorflow as tf
        tf_gpus = tf.config.list_physical_devices('GPU')
        print("\n[TensorFlow]")
        print(f"  GPUs Detected: {len(tf_gpus)}")
        for g in tf_gpus:
            print(f"  Device: {g}")
        print(f"  Mixed Precision Policy: {tf.keras.mixed_precision.global_policy().name}")
    except Exception as e:
        print(f"\n[TensorFlow] Error: {e}")

    # XGBoost
    try:
        import xgboost as xgb
        print("\n[XGBoost]")
        print(f"  Version: {xgb.__version__}")
        print(f"  GPU Params: {get_xgboost_gpu_params()}")
    except Exception as e:
        print(f"\n[XGBoost] Error: {e}")

    print("\n" + "=" * 60)
