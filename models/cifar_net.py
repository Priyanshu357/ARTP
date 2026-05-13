import onnx

# Load the ONNX model
onnx_model = onnx.load(".\adversarial-platform\models\cifar_net.onnx")

# Check if the model is valid
onnx.checker.check_model(onnx_model)

print("ONNX model loaded and checked successfully!")
print(f"The model has {len(onnx_model.graph.input)} input(s).")
for i, input_node in enumerate(onnx_model.graph.input):
    print(f"Input {i}: Name = {input_node.name}, Shape = {[dim.dim_value for dim in input_node.type.tensor_type.shape.dim]}")

print(f"The model has {len(onnx_model.graph.output)} output(s).")
for i, output_node in enumerate(onnx_model.graph.output):
    print(f"Output {i}: Name = {output_node.name}, Shape = {[dim.dim_value for dim in output_node.type.tensor_type.shape.dim]}")

print(f"ONNX Opset Version: {onnx_model.opset_import[0].version}")