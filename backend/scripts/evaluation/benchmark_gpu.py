import time

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from src.utils.gpu_utils import (
    get_catboost_gpu_params,
    get_lightgbm_gpu_params,
    get_xgboost_gpu_params,
)


def benchmark_pytorch(X, y):
    print("\n--- Benchmarking PyTorch (Simple MLP) ---")
    model = nn.Sequential(
        nn.Linear(X.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )
    criterion = nn.CrossEntropyLoss()

    # CPU Benchmark
    model.cpu()
    X_cpu, y_cpu = torch.FloatTensor(X), torch.LongTensor(y)
    optimizer = torch.optim.Adam(model.parameters())

    start = time.perf_counter()
    for _ in range(50):
        optimizer.zero_grad()
        out = model(X_cpu)
        loss = criterion(out, y_cpu)
        loss.backward()
        optimizer.step()
    cpu_time = time.perf_counter() - start
    print(f"PyTorch CPU Time: {cpu_time:.4f}s")

    # GPU Benchmark
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark.")
        return

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    model.cuda()
    X_gpu, y_gpu = X_cpu.cuda(), y_cpu.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()

    start = time.perf_counter()
    for _ in range(50):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(X_gpu)
            loss = criterion(out, y_gpu)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"PyTorch GPU Time: {gpu_time:.4f}s")
    print(f"Speedup Factor: {cpu_time / gpu_time:.2f}x")
    print(f"Peak VRAM: {peak_vram:.2f} MB")

def benchmark_xgboost(X, y):
    print("\n--- Benchmarking XGBoost ---")
    # CPU
    params_cpu = {"tree_method": "hist", "device": "cpu", "objective": "multi:softprob", "num_class": 3, "n_estimators": 50}
    start = time.perf_counter()
    model = xgb.XGBClassifier(**params_cpu)
    model.fit(X, y)
    cpu_time = time.perf_counter() - start
    print(f"XGBoost CPU Time: {cpu_time:.4f}s")

    # GPU
    if not torch.cuda.is_available():
        return

    params_gpu = {"objective": "multi:softprob", "num_class": 3, "n_estimators": 50, **get_xgboost_gpu_params()}
    start = time.perf_counter()
    model = xgb.XGBClassifier(**params_gpu)
    model.fit(X, y)
    gpu_time = time.perf_counter() - start

    print(f"XGBoost GPU Time: {gpu_time:.4f}s")
    print(f"Speedup Factor: {cpu_time / gpu_time:.2f}x")

def benchmark_lightgbm(X, y):
    print("\n--- Benchmarking LightGBM ---")
    # CPU
    start = time.perf_counter()
    model = LGBMClassifier(n_estimators=50, device="cpu", verbose=-1)
    model.fit(X, y)
    cpu_time = time.perf_counter() - start
    print(f"LightGBM CPU Time: {cpu_time:.4f}s")

    # GPU
    if not torch.cuda.is_available():
        return

    params_gpu = get_lightgbm_gpu_params()
    if not params_gpu:
        print("LightGBM GPU params not available.")
        return

    start = time.perf_counter()
    model = LGBMClassifier(n_estimators=50, verbose=-1, **params_gpu)
    model.fit(X, y)
    gpu_time = time.perf_counter() - start

    print(f"LightGBM GPU Time: {gpu_time:.4f}s")
    print(f"Speedup Factor: {cpu_time / gpu_time:.2f}x")

def benchmark_catboost(X, y):
    print("\n--- Benchmarking CatBoost ---")
    # CPU
    start = time.perf_counter()
    model = CatBoostClassifier(iterations=50, task_type="CPU", verbose=0)
    model.fit(X, y)
    cpu_time = time.perf_counter() - start
    print(f"CatBoost CPU Time: {cpu_time:.4f}s")

    # GPU
    if not torch.cuda.is_available():
        return

    start = time.perf_counter()
    model = CatBoostClassifier(iterations=50, verbose=0, **get_catboost_gpu_params())
    model.fit(X, y)
    gpu_time = time.perf_counter() - start

    print(f"CatBoost GPU Time: {gpu_time:.4f}s")
    print(f"Speedup Factor: {cpu_time / gpu_time:.2f}x")

def main():
    print("Generating synthetic data for benchmarking...")
    np.random.seed(42)
    X = np.random.randn(50000, 100).astype(np.float32)
    y = np.random.randint(0, 3, size=(50000,))

    benchmark_pytorch(X, y)
    benchmark_xgboost(X, y)
    benchmark_lightgbm(X, y)
    benchmark_catboost(X, y)

if __name__ == "__main__":
    main()
