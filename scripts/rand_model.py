import torch
import torchvision.models as models

# Load a pretrained ResNet18
model = models.resnet18(pretrained=True)
model.eval()

# Dummy input matching model input shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
