import argparse
import time
import numpy as np
import onnx
import onnxruntime as ort
import os
import csv
import torch
import platform
import subprocess
import pyRAPL

# ------------------------
# Energy / Power setup
# ------------------------
pyRAPL.setup()

# ------------------------
# Device info
# ------------------------
def get_cpu_name():
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.strip().split(":")[1].strip()
        return platform.processor()
    except:
        return "Unknown CPU"

def get_gpu_name():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        )
        return output.strip().split("\n")[0]
    except Exception:
        return "No GPU"

# ------------------------
# Dummy input
# ------------------------
def generate_dummy_input(model, dtype="float32"):
    if model.graph.input:
        input_tensor = model.graph.input[0]
        shape = [dim.dim_value if dim.dim_value > 0 else 1
                 for dim in input_tensor.type.tensor_type.shape.dim]
    else:
        shape = [1, 3, 224, 224]
    np_dtype = np.float32 if dtype == "float32" else np.float16
    return np.random.randn(*shape).astype(np_dtype), tuple(shape)

# ------------------------
# CPU profiling using pyRAPL
# ------------------------
def measure_cpu_op(run_fn, runs=10):
    durations, energies = [], []
    for _ in range(runs):
        start = time.time()
        meter = pyRAPL.Measurement("cpu_op")
        meter.begin()
        run_fn()
        meter.end()
        end = time.time()
        durations.append(end - start)
        # Access first CPU package energy in µJ and convert to J
        energies.append(meter.result.pkg[0] / 1e6)
    avg_duration = np.mean(durations)
    avg_energy = np.mean(energies)
    avg_power = avg_energy / avg_duration if avg_duration > 0 else 0
    return avg_duration, avg_energy, avg_power

# ------------------------
# GPU profiling using nvidia-smi
# ------------------------
def measure_gpu_op(run_fn, runs=10):
    durations, powers = [], []
    for _ in range(runs):
        # GPU power before
        start_power = float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip())
        start = time.time()
        run_fn()
        end = time.time()
        # GPU power after
        end_power = float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip())
        durations.append(end - start)
        powers.append((start_power + end_power) / 2.0)
    avg_duration = np.mean(durations)
    avg_power = np.mean(powers)
    avg_energy = avg_power * avg_duration
    return avg_duration, avg_energy, avg_power

# ------------------------
# Profile a single op
# ------------------------
def profile_op_node(op_name, op_type, input_tensor, device, device_name, runs=10):
    providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]

    def run_fn():
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession("model.onnx", sess_options, providers=providers)
        input_name = sess.get_inputs()[0].name
        sess.run(None, {input_name: input_tensor})
        del sess
        if device == "cuda":
            torch.cuda.empty_cache()

    if device == "cpu":
        duration, energy, power = measure_cpu_op(run_fn, runs=runs)
    else:
        duration, energy, power = measure_gpu_op(run_fn, runs=runs)

    return {
        "op_name": op_name,
        "op_type": op_type,
        "input_shape": str(input_tensor.shape),
        "device": device,
        "device_name": device_name,
        "avg_duration_s": duration,
        "avg_power_w": power,
        "avg_energy_j": energy
    }

# ------------------------
# Profile full network
# ------------------------
def profile_network_per_op(onnx_model_path, device, runs=10, csv_file="network_per_op_results.csv", dtype="float32"):
    model = onnx.load(onnx_model_path)
    results = []

    device_name = get_cpu_name() if device == "cpu" else get_gpu_name()
    print(f"Profiling on device: {device} ({device_name})")

    input_tensor, _ = generate_dummy_input(model, dtype=dtype)

    for node in model.graph.node:
        op_name = node.name or node.op_type
        op_type = node.op_type
        profile = profile_op_node(op_name, op_type, input_tensor, device, device_name, runs=runs)
        results.append(profile)
        print(
            f"Profiled {op_name} ({op_type}) on {device_name}: "
            f"duration={profile['avg_duration_s']:.6f}s, "
            f"energy={profile['avg_energy_j']:.6f}J, "
            f"power={profile['avg_power_w']:.2f}W"
        )

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
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per op")
    parser.add_argument("--csv", default="network_per_op_results.csv", help="CSV output file")
    parser.add_argument("--dtype", default="float32", help="Tensor precision")
    args = parser.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    profile_network_per_op(args.model, device, runs=args.runs, csv_file=args.csv, dtype=args.dtype)

if __name__ == "__main__":
    main()
