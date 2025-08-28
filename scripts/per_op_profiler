## command to run:python per_op_profiler.py --op relu --input-shape 1 3 224 224 --dtype float32 --device cuda --runs 10 --csv results.csv
##!pip install torch pyJoules numpy
##!sudo apt install msr-tools
##!sudo modprobe msr


##-----------imports and file writing-----------------
#import pandas as pd
# Load CSV
#df = pd.read_csv("results.csv")
# Show first 5 rows
#df.head()

# per_op_profiler.py
import argparse
import time
import csv
import torch
import subprocess
import platform
import os

# ------------------------
# Energy measurement helpers
# ------------------------
def measure_gpu_energy_nvidia_smi(interval=0.05, duration=1.0, device_index=0):
    """Return average power (W) and energy (J) for GPU using nvidia-smi"""
    powers = []
    n_samples = int(duration / interval)
    for _ in range(n_samples):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 f"--query-gpu=power.draw",
                 "--format=csv,noheader,nounits",
                 f"--id={device_index}"],
                stderr=subprocess.DEVNULL
            )
            power_w = float(out.decode().strip())
            powers.append(power_w)
        except Exception:
            pass
        time.sleep(interval)
    avg_power = sum(powers)/len(powers) if powers else 0.0
    energy = avg_power * duration
    return avg_power, energy

# CPU energy via RAPL (Linux only)
def measure_cpu_energy(duration=1.0):
    try:
        # Using pyRAPL if available
        import pyRAPL
        meter = pyRAPL.Measurement('tmp')
        pyRAPL.setup()
        meter.begin()
        time.sleep(duration)
        meter.end()
        # energy in microjoules -> joules
        energy_j = meter.result.pkg[0] * 1e-6 if meter.result.pkg else 0.0
        return None, energy_j
    except Exception:
        return None, None

# ------------------------
# Device info
# ------------------------
def get_device_info(device):
    if device.type == "cuda":
        return "GPU", torch.cuda.get_device_name(device)
    else:
        return "CPU", platform.processor() or platform.machine()

# ------------------------
# Profiling per op
# ------------------------
def profile_op(op_name, input_shape, dtype_str, device, runs=10):
    # Map string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.float32)

    # Generate input
    tensor = torch.randn(*input_shape, dtype=dtype, device=device)

    # Select operation
    if op_name.lower() == "relu":
        op = torch.nn.functional.relu
    elif op_name.lower() == "sigmoid":
        op = torch.sigmoid
    elif op_name.lower() == "tanh":
        op = torch.tanh
    elif op_name.lower() == "matmul":
        # For matmul, second tensor
        tensor2 = torch.randn(*input_shape, dtype=dtype, device=device)
        op = lambda x: torch.matmul(x, tensor2)
    else:
        raise ValueError(f"Unsupported op: {op_name}")

    # Warm-up
    for _ in range(3):
        out = op(tensor)

    # Measure time + energy
    times = []
    if device.type == "cuda":
        torch.cuda.synchronize()
        avg_power, total_energy = measure_gpu_energy_nvidia_smi(duration=1.0, device_index=device.index if hasattr(device,'index') else 0)
    else:
        avg_power, total_energy = measure_cpu_energy(duration=1.0)

    for _ in range(runs):
        t0 = time.time()
        out = op(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0)*1000.0)  # ms

    avg_time_ms = sum(times)/len(times)
    device_type, device_name = get_device_info(device)
    return {
        "op_type": op_name,
        "input_shape": input_shape,
        "device_type": device_type,
        "device_name": device_name,
        "precision": dtype_str,
        "avg_time_ms": avg_time_ms,
        "avg_power_w": avg_power,
        "total_energy_j": total_energy
    }

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, help="Operation to profile (relu, sigmoid, tanh, matmul)")
    parser.add_argument("--input-shape", type=int, nargs="+", required=True, help="Input shape e.g., 1 3 224 224")
    parser.add_argument("--dtype", default="float32", help="Tensor precision")
    parser.add_argument("--device", default=None, choices=["cpu","cuda"], help="Device to run on")
    parser.add_argument("--runs", type=int, default=10, help="Number of iterations")
    parser.add_argument("--csv", default="profiler_results.csv", help="CSV output file")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    result = profile_op(args.op, tuple(args.input_shape), args.dtype, device, args.runs)

    # Save CSV
    file_exists = os.path.isfile(args.csv)
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print("Profiling complete. Results:")
    print(result)

if __name__ == "__main__":
    main()

