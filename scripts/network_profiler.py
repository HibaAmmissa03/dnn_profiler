# network_per_op_profiler_fixed.py

import argparse
import time
import numpy as np
import onnx
import onnxruntime as ort
import os
import csv
import torch
import platform

# Optional energy libraries
try:
    import pyRAPL
    pyRAPL.setup()
    has_pyrapl = True
except:
    has_pyrapl = False
    import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    has_gpu = True
except:
    has_gpu = False

# ------------------------
# Get full device names
# ------------------------
def get_cpu_name():
    try:
        if platform.system() == "Windows":
            try:
                import wmi
                w = wmi.WMI()
                return w.Win32_Processor()[0].Name
            except:
                pass
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.strip().split(":")[1].strip()
        return platform.processor()
    except:
        return "Unknown CPU"

def get_gpu_name(gpu_index=0):
    if has_gpu:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return pynvml.nvmlDeviceGetName(handle).decode("utf-8")
    return "No GPU"

# ------------------------
# Generate dummy input
# ------------------------
def generate_dummy_input(model, dtype="float32"):
    if model.graph.input:
        input_tensor = model.graph.input[0]
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in input_tensor.type.tensor_type.shape.dim]
    else:
        shape = [1, 3, 224, 224]  # fallback
    np_dtype = np.float32 if dtype=="float32" else np.float16
    return np.random.randn(*shape).astype(np_dtype), tuple(shape)

# ------------------------
# CPU energy measurement
# ------------------------
def measure_cpu_energy():
    if has_pyrapl:
        meter = pyRAPL.Measurement("run")
        meter.begin()
        return meter
    else:
        return None

def finalize_cpu_energy(meter, duration_s):
    if has_pyrapl and meter:
        meter.end()
        return meter.result.pkg[0]
    else:
        # fallback: approximate energy using CPU percent
        cpu_percent = psutil.cpu_percent(interval=None)
        max_power_w = 65  # typical CPU max power
        return max_power_w * cpu_percent / 100 * duration_s

# ------------------------
# GPU power measurement
# ------------------------
def measure_gpu_power(gpu_index=0):
    if has_gpu:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
    return 0.0

# ------------------------
# Profile a single op
# ------------------------
def profile_op_node(node_name, node_type, input_tensor, device, device_name, runs=3):
    """
    Memory-optimized per-op profiling using ORT session per run.
    """
    duration_list = []
    power_list = []

    providers = ["CPUExecutionProvider"] if device=="cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]

    for _ in range(runs):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # reduce memory usage
        sess_options.enable_profiling = False

        sess = ort.InferenceSession("model.onnx", sess_options, providers=providers)
        input_name = sess.get_inputs()[0].name

        start = time.time()
        if device=="cpu":
            meter = measure_cpu_energy()
        else:
            gpu_power_before = measure_gpu_power()

        sess.run(None, {input_name: input_tensor})
        end = time.time()
        duration_s = end - start

        if device=="cpu":
            energy_j = finalize_cpu_energy(meter, duration_s)
            avg_power = energy_j / duration_s
        else:
            gpu_power_after = measure_gpu_power()
            avg_power = (gpu_power_before + gpu_power_after)/2
            energy_j = avg_power * duration_s

        duration_list.append(duration_s)
        power_list.append(avg_power)

        # Delete session to free memory
        del sess
        torch.cuda.empty_cache() if device=="cuda" else None

    avg_duration = np.mean(duration_list)
    avg_power = np.mean(power_list)

    return {
        "op_name": node_name,
        "input_shape": str(input_tensor.shape),
        "device": device,
        "device_type": device_name,
        "avg_duration_s": avg_duration,
        "avg_power_w": avg_power,
        "energy_j": avg_power * avg_duration
    }

# ------------------------
# Profile network per op
# ------------------------
def profile_network_per_op(onnx_model_path, device, runs=3, csv_file="network_per_op_results.csv", dtype="float32"):
    model = onnx.load(onnx_model_path)
    results = []

    device_name = get_cpu_name() if device=="cpu" else get_gpu_name()
    print(f"Profiling on device: {device} ({device_name})")

    if device=="cpu":
        if has_pyrapl:
            print("Using pyRAPL for CPU energy measurement")
        else:
            print("Using psutil as fallback for CPU energy measurement")
    elif device=="cuda":
        print(f"Using pynvml for GPU power measurement: {device_name}" if has_gpu else "GPU not detected, no power measurement")

    input_tensor, _ = generate_dummy_input(model, dtype=dtype)

    for node in model.graph.node:
        op_name = node.name or node.op_type
        op_type = node.op_type
        profile = profile_op_node(op_name, op_type, input_tensor, device, device_name, runs=runs)
        results.append(profile)
        print(f"Profiled {op_name} ({op_type}) on {device_name}: duration={profile['avg_duration_s']:.6f}s, energy={profile['energy_j']:.3f}J")

    # Save CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Profiling complete. Results saved to {csv_file}")
    return results

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per op")
    parser.add_argument("--csv", default="network_per_op_results.csv", help="CSV output file")
    parser.add_argument("--dtype", default="float32", help="Tensor precision")
    args = parser.parse_args()

    device = "cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu"
    profile_network_per_op(args.model, device, runs=args.runs, csv_file=args.csv, dtype=args.dtype)

if __name__ == "__main__":
    main()
