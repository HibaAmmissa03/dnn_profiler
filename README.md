# Deep Neural Network (DNN) Profiler

This project provides a set of tools for profiling the execution time and energy consumption of deep learning operations and ONNX models.
It supports both CPU and GPU (CUDA) backends and allows profiling either at the per-operation level or across all operators in a full ONNX model.

## 📂 Repository Structure

**-network_profiler.py**
Loads an ONNX model and runs per-operation profiling for each node in the computational graph.
Results are aggregated and exported to CSV.

**-op_parser.py**
Utility script to inspect an ONNX model. Prints model metadata, input/output details, and performs a test inference using ONNX Runtime.

**-rand_model.py, or equivalent**
Initital version: Exports a pretrained ResNet18 model from PyTorch to ONNX format for testing.
Now, after a successful initial first run with this sample model, we have added multiple pretrained models categorized as follows: mobile models, large models, and random CNNs which are all fed the same dummy input for the sake of comparison. 


**-requirements.txt**
Python package dependencies required to run the profilers.

**-network_profile.csv**
Example output CSV showing profiling results from network_profiler.py.


## ⚙️ Installation

**1. Clone the repository:** git clone https://github.com/HibaAmmissa03/dnn_profiler
cd dnn-profiler

**2. Install Python dependencies:**
pip install -r requirements.txt

**3. (Linux only) Install CPU energy measurement tools:**
sudo apt install msr-tools
sudo modprobe msr

## 🚀 Usage

### Network Profiler (network_profiler.py)

Profiles all operations in an ONNX model using per_op_profiler internally.
Results are written to a CSV file with per-operator statistics.

**Commands to run:**
In a virtual environment with all the requirements (requirements.txt) already installed, and on a linux machine with root access to intel rapl (Running Average Power Limit) run the following commands:
sudo chmod -R a+r /sys/class/powercap/intel-rapl
sudo .venv/bin/python3 network_profiler.py --model densenet201.onnx --model_name densenet201 --runs 10 --device cpu --csv /home/hibz/dnn_profiler/scripts/network_profile.csv

**Arguments:**
--model : Path to ONNX model file
--device : Execution device (cpu or cuda)
--runs : Number of repetitions for averaging (default: 10)
--csv : Output CSV file (default: network_per_op_results.csv)
--dtype : Precision for tensors (default: float32)

## 📊 Example Workflow:

The project consists of three main scripts that work together in a pipeline:

### 1. rand_model.py (or any other variation)

-Loads a pretrained PyTorch model (e.g., ResNet18).

-Converts the model to ONNX format using torch.onnx.export.

-Saves the exported ONNX model as model.onnx.

**Output:**
model.onnx – the ONNX representation of the PyTorch model.

### 2. op_parser.py 

-Loads the ONNX model from disk.

-Validates the model using onnx.checker.

-Prints key information: IR version and producer name, Number of nodes in the computation graph, Input and output names, shapes, and types.

-Runs a dummy inference with ONNX Runtime to confirm the model executes correctly.

**Purpose:** 
Ensures that the exported model is valid and produces expected outputs.

### 3. network_profiler.py 

-Loads the ONNX model and profiles it operator by operator.

-Measures for each op: Execution time (average over multiple runs), Energy and power consumption (CPU via [pyRAPL], GPU via nvidia-smi), Supports both CPU and CUDA GPU execution providers, Saves results into a CSV file for further analysis.

**Output:**
A CSV file (e.g., network_profile.csv) with detailed per-op profiling results.


## 📦 Requirements: All dependencies are listed in requirements.txt:

torch>=2.0.0
onnx>=1.15.0
onnxruntime>=1.15.0
numpy>=1.25.0
pandas>=2.1.0
pyRAPL>=0.12.0

pip install -r requirements.txt
   
