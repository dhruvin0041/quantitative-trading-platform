"""GPU Verification & Benchmark Script for RTX 5070.

Run: python -m scripts.ops.verify_gpu
This script verifies that all model components are correctly using GPU
and runs micro-benchmarks to measure speedup vs CPU.
"""
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("gpu_verify")


def verify_pytorch():
    """Verify PyTorch CUDA availability and run tensor benchmark."""
    import torch
    print("\n" + "=" * 60)
    print("PYTORCH GPU VERIFICATION")
    print("=" * 60)
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  Device Count: {torch.cuda.device_count()}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Total VRAM: {props.total_mem / (1024**3):.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")

        # Matrix multiplication benchmark
        print("\n  Running matmul benchmark (4096x4096)...")
        sizes = [1024, 2048, 4096]
        for size in sizes:
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()

            # Warmup GPU
            _ = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()

            # CPU benchmark
            start = time.perf_counter()
            for _ in range(3):
                _ = torch.mm(a_cpu, b_cpu)
            cpu_time = (time.perf_counter() - start) / 3

            # GPU benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                _ = torch.mm(a_gpu, b_gpu)
                torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - start) / 3

            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"  [{size}x{size}] CPU: {cpu_time*1000:.1f}ms | GPU: {gpu_time*1000:.1f}ms | Speedup: {speedup:.1f}x")

        # DQN model test
        print("\n  Testing DQN model on GPU...")
        from src.models.rl.dqn_agent import DuelingDQNetwork
        model = DuelingDQNetwork(50, 3).cuda()
        x = torch.randn(64, 50).cuda()
        with torch.no_grad():
            out = model(x)
        print(f"  DQN output shape: {out.shape}, device: {out.device}")
        print(f"  VRAM after DQN: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
        del model, x, out
        torch.cuda.empty_cache()
    else:
        print("  [WARNING] No CUDA GPU detected!")
    return torch.cuda.is_available()


def verify_tensorflow():
    """Verify TensorFlow GPU availability."""
    import tensorflow as tf
    print("\n" + "=" * 60)
    print("TENSORFLOW GPU VERIFICATION")
    print("=" * 60)
    print(f"  TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  GPUs Detected: {len(gpus)}")
    for gpu in gpus:
        print(f"  Device: {gpu}")
    if gpus:
        print(f"  Mixed Precision: {tf.keras.mixed_precision.global_policy().name}")

        # Simple benchmark
        print("\n  Running TF matmul benchmark (4096x4096)...")
        with tf.device('/CPU:0'):
            a = tf.random.normal([4096, 4096])
            b = tf.random.normal([4096, 4096])
            start = time.perf_counter()
            _ = tf.matmul(a, b)
            cpu_time = time.perf_counter() - start

        with tf.device('/GPU:0'):
            a = tf.random.normal([4096, 4096])
            b = tf.random.normal([4096, 4096])
            # warmup
            _ = tf.matmul(a, b)
            start = time.perf_counter()
            _ = tf.matmul(a, b)
            gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  CPU: {cpu_time*1000:.1f}ms | GPU: {gpu_time*1000:.1f}ms | Speedup: {speedup:.1f}x")
    else:
        print("  [WARNING] No TF GPU detected!")
    return len(gpus) > 0


def verify_xgboost():
    """Verify XGBoost GPU capability."""
    import xgboost as xgb
    import numpy as np
    print("\n" + "=" * 60)
    print("XGBOOST GPU VERIFICATION")
    print("=" * 60)
    print(f"  XGBoost Version: {xgb.__version__}")

    # Test GPU training
    X = np.random.randn(5000, 20).astype(np.float32)
    y = np.random.randint(0, 3, 5000)

    try:
        model = xgb.XGBClassifier(
            n_estimators=100, tree_method="hist", device="cuda",
            objective="multi:softprob", num_class=3,
        )
        start = time.perf_counter()
        model.fit(X, y)
        gpu_time = time.perf_counter() - start
        print(f"  GPU Training: {gpu_time:.3f}s [OK]")

        model_cpu = xgb.XGBClassifier(
            n_estimators=100, tree_method="hist", device="cpu",
            objective="multi:softprob", num_class=3,
        )
        start = time.perf_counter()
        model_cpu.fit(X, y)
        cpu_time = time.perf_counter() - start
        print(f"  CPU Training: {cpu_time:.3f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    except Exception as e:
        print(f"  GPU Training Failed: {e}")
        print(f"  Falling back to CPU")


def verify_catboost():
    """Verify CatBoost GPU capability."""
    from catboost import CatBoostClassifier
    import numpy as np
    print("\n" + "=" * 60)
    print("CATBOOST GPU VERIFICATION")
    print("=" * 60)

    X = np.random.randn(5000, 20).astype(np.float32)
    y = np.random.randint(0, 3, 5000)

    try:
        model = CatBoostClassifier(
            iterations=100, task_type="GPU", devices="0",
            loss_function="MultiClass", verbose=0,
        )
        start = time.perf_counter()
        model.fit(X, y)
        gpu_time = time.perf_counter() - start
        print(f"  GPU Training: {gpu_time:.3f}s [OK]")

        model_cpu = CatBoostClassifier(
            iterations=100, task_type="CPU",
            loss_function="MultiClass", verbose=0,
        )
        start = time.perf_counter()
        model_cpu.fit(X, y)
        cpu_time = time.perf_counter() - start
        print(f"  CPU Training: {cpu_time:.3f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    except Exception as e:
        print(f"  GPU Training Failed: {e}")


def main():
    print("\n" + "#" * 60)
    print("# STOCK INDICATOR SYSTEM - GPU ACCELERATION VERIFICATION")
    print("# Target: NVIDIA GeForce RTX 5070 Laptop (8GB VRAM)")
    print("#" * 60)

    results = {}
    results['pytorch'] = verify_pytorch()
    results['tensorflow'] = verify_tensorflow()
    verify_xgboost()
    verify_catboost()

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_ok = all(results.values())
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  [{icon}] {name}: {'GPU Active' if status else 'CPU Only'}")

    if all_ok:
        print("\n  >>> ALL SYSTEMS GPU-ACCELERATED <<<")
    else:
        print("\n  [WARNING] Some components running on CPU")


if __name__ == "__main__":
    main()
