import onnx
import onnxruntime as ort
import numpy as np

# Step 1: Load the ONNX model from file
model_path = "C:/Users/PC/onnx_trial/model.onnx"
onnx_model = onnx.load(model_path)

# Step 2: Check that the model is valid
onnx.checker.check_model(onnx_model)

# Step 3: Print some basic info about the graph
print("Model IR version:", onnx_model.ir_version)
print("Model producer:", onnx_model.producer_name)
print("Number of nodes in graph:", len(onnx_model.graph.node))
print("Inputs:")
for i in onnx_model.graph.input:
    print(f"  - {i.name}")

print("Outputs:")
for o in onnx_model.graph.output:
    print(f"  - {o.name}")

# Step 4: Create an ONNX Runtime inference session
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Step 5: Get input name and shape from the session
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print("First input name:", input_name)
print("Shape:", input_shape)
print("Type:", input_type)

# Step 6: Generate dummy data 
# if the input shape is [1, 3, 224, 224] (image with RGB channels),
# we create random data with the same shape.
dummy_input = np.random.randn(*[d if isinstance(d, int) else 1 for d in input_shape]).astype(np.float32)

# Step 7: Run inference
outputs = session.run(None, {input_name: dummy_input})

# Step 8: Print inference results
print("Number of outputs:", len(outputs))
for idx, out in enumerate(outputs):
    print(f"Output {idx} shape:", out.shape)
    print(f"Output {idx} (first 5 values):", out.flatten()[:5])
