import onnx
from onnx_coreml import convert

# Load the ONNX model
onnx_model = onnx.load("OULU_Protocol_2_model_0_0.onnx")

# Convert the ONNX model to Core ML
coreml_model = convert(onnx_model)

# Save the Core ML model
coreml_model.save("OULU_Protocol_2_model.mlmodel")
print("Model converted and saved as 'OULU_Protocol_2_model.mlmodel'")
