import torch
from tf2onnx import convert

# Carga tu modelo entrenado en PyTorch
model = torch.load('D:\ProyectoBotellas\yoloV8')

# Convierte el modelo a formato ONNX
onnx_model = convert.from_pytorch(model, input_names=['input'], opset=11)

# Guarda el modelo en formato ONNX
onnx_path = 'modelo_yolov5.onnx'
with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())
