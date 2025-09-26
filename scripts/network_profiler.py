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

# ------------------------
# Optional CPU energy library
# ------------------------
try:
    import pyRAPL
    pyRAPL.setup()
    has_pyrapl = True
except:
    has_pyrapl = False

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
                return platform.processor()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.strip().split(":")[1].strip()
        return platform.processor()
    except:
        return "Unknown CPU"

def get_gpu_name():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        return output.decode("utf-8").strip()
    except:
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
# Measure CPU energy for a single run
# ------------------------
def measure_cpu_op(run_fn, runs=10):
    durations = []
    energies = []

    for _ in range(runs):
        if has_pyrapl:
            meter = pyRAPL.Measurement("run")
            meter.begin()
        start = time.time()
        run_fn()
        end = time.time()
        duration = end - start
        durations.append(duration)
        if has_pyrapl:
            meter.end()
            # pyRAPL returns package energy in microjoules
            energy_j = meter.result.pkg[0] / 1e6
        else:
            energy_j = 0.0
        energies.append(energy_j)
    avg_duration = np.mean(durations)
    avg_energy = np.mean(energies)
    avg_power = avg_energy / avg_duration if avg_duration > 0 else 0.0
    return avg_duration, avg_energy, avg_power

# ------------------------
# Measure GPU power using nvidia-smi
# ------------------------
def measure_gpu_power():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        )
        return float(output.decode("utf-8").strip())
    except:
        return 0.0

# ------------------------
# Profile a single op
# ------------------------
def profile_op_node(model_path, node_name, node_type, input_tensor, device, device_name, runs=10):
    providers = ["CPUExecutionProvider"] if device=="cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run_fn():
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.enable_profiling = False
        sess = ort.InferenceSession(model_path, sess_options, providers=providers)
        input_name = sess.get_inputs()[0].name
        sess.run(None, {input_name: input_tensor})
        del sess
        torch.cuda.empty_cache() if device=="cuda" else None

    if device == "cpu":
        avg_duration, avg_energy, avg_power = measure_cpu_op(run_fn, runs=runs)
    else:
        durations = []
        powers = []
        for _ in range(runs):
            start = time.time()
            run_fn()
            end = time.time()
            durations.append(end - start)
            powers.append(measure_gpu_power())
        avg_duration = np.mean(durations)
        avg_power = np.mean(powers)
        avg_energy = avg_power * avg_duration

    return {
        "op_name": node_name,
        "op_type": node_type,
        "input_shape": str(input_tensor.shape),
        "device": device,
        "device_name": device_name,
        "avg_duration_s": avg_duration,
        "avg_power_w": avg_power,
        "avg_energy_j": avg_energy
    }

# ------------------------
# Profile network per op
# ------------------------
def profile_network_per_op(onnx_model_path, model_name, device, runs=10, csv_file="network_per_op_results.csv", dtype="float32"):
    model = onnx.load(onnx_model_path)
    results = []

    device_name = get_cpu_name() if device=="cpu" else get_gpu_name()
    print(f"Profiling model {model_name} on device: {device} ({device_name})")

    input_tensor, _ = generate_dummy_input(model, dtype=dtype)

    for node in model.graph.node:
        op_name = node.name or node.op_type
        op_type = node.op_type
        profile = profile_op_node(onnx_model_path, op_name, op_type, input_tensor, device, device_name, runs=runs)
        profile["model_name"] = model_name
        results.append(profile)
        print(f"Profiled {op_name} ({op_type}): duration={profile['avg_duration_s']:.6f}s, power={profile['avg_power_w']:.3f}W")

    # Save CSV
    with open(csv_file, "a", newline="") as f:
        fieldnames = ["model_name","op_name","op_type","input_shape","device","device_name","avg_duration_s","avg_power_w","avg_energy_j"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    parser.add_argument("--model_name", required=True, help="Name of the model being profiled")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per op")
    parser.add_argument("--csv", default="network_per_op_results.csv", help="CSV output file")
    parser.add_argument("--dtype", default="float32", help="Tensor precision")
    args = parser.parse_args()

    device = "cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu"
    profile_network_per_op(args.model, args.model_name, device, runs=args.runs, csv_file=args.csv, dtype=args.dtype)

if __name__ == "__main__":
    main()
