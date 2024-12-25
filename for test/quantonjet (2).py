import warnings
warnings.filterwarnings('ignore')  # Ignore warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import cv2
import time
val_path = 'images/validation/'
# Loading validation data
x_val = []
y_val = []
for class_idx, class_name in enumerate(os.listdir(val_path)):
    class_folder = os.path.join(val_path, class_name)
    for img_file in os.listdir(class_folder):
        img = cv2.imread(os.path.join(class_folder, img_file))
        img = cv2.resize(img, (48, 48))
        x_val.append(img)
        y_val.append(class_idx)

x_val = np.array(x_val)
y_val = np.array(y_val)

y_val_tensor = torch.tensor(y_val, dtype=torch.long)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)


# 모델 정의
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


#Reference https://github.com/StevenHSKim/FER-Jetson-Nano/blob/main/training/PTQ_KD_training.py
class WeightQuantizer:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        if num_bits == 32:
            self.qmin, self.qmax = float('-inf'), float('inf')
        else:
            self.qmin, self.qmax = -(2 ** (num_bits - 1)), (2 ** (num_bits - 1)) - 1
        self.scale = None
        self.zero_point = 0  # Symmetric quantization uses 0

    def calibrate(self, tensor):
        """Compute quantization parameters based on the given tensor."""
        if self.num_bits == 32:
            self.scale = 1.0
            return
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        max_abs = max(abs(max_val), abs(min_val))
        self.scale = max_abs / (self.qmax)
        self.scale = max(self.scale, 1e-8)  # Prevent zero-scale

    def quantize(self, tensor):
        """Quantize the given tensor using computed parameters."""
        if self.scale is None:
            self.calibrate(tensor)

        if self.num_bits == 32:
            return tensor

        quantized = torch.round(tensor / self.scale).clamp(self.qmin, self.qmax)
        dtype_map = {4: torch.int8, 8: torch.int8, 16: torch.int16, 32: torch.float32}
        return quantized.to(dtype_map[self.num_bits])

    def dequantize(self, quantized_tensor):
        """Dequantize the tensor back to its original range."""
        if self.num_bits == 32:
            return quantized_tensor
        return quantized_tensor.float() * self.scale


class QuantizedLayer:
    def __init__(self, weight_tensor, num_bits):
        self.quantizer = WeightQuantizer(num_bits)
        self.quantized_weight = self.quantizer.quantize(weight_tensor)

    def get_dequantized_weight(self):
        """Retrieve the dequantized weight."""
        return self.quantizer.dequantize(self.quantized_weight)


class QuantizedModel(nn.Module):
    def __init__(self, original_model, num_bits):
        super().__init__()
        self.original_model = original_model
        self.num_bits = num_bits
        self.quantized_layers = {}

        self._quantize_model()

    def _quantize_model(self):
        """Apply quantization to all layers with weights."""
        for name, module in self.original_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                quantized_layer = QuantizedLayer(module.weight.data.clone(), self.num_bits)
                self.quantized_layers[name] = quantized_layer
                module.weight.data.copy_(quantized_layer.get_dequantized_weight())

    def forward(self, x):
        return self.original_model(x)

    def estimated_size(self):
        """Estimate the memory size of the quantized model."""
        param_count = sum(
            module.weight.numel()
            for module in self.original_model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        )
        param_size = 1 if self.num_bits <= 8 else 2
        return param_count * param_size + 1024  # Add 1 KB for metadata

    def save_model(self, filepath):
        """Save the quantized model and parameters."""
        save_data = {
            'num_bits': self.num_bits,
            'state_dict': {},
            'quantization_params': {}
        }

        for name, qlayer in self.quantized_layers.items():
            save_data['state_dict'][name] = qlayer.quantized_weight
            save_data['quantization_params'][name] = {
                'scale': qlayer.quantizer.scale,
                'zero_point': qlayer.quantizer.zero_point
            }

        torch.save(save_data, filepath)

acc = []
predic_time = []
Macs = []
size = []


for i in [4,8,16,32]:
    # 모델 생성 및 양자화
    original_model = EmotionRecognitionModel()
    #%%
    original_model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location=torch.device('cpu')))

    num_bits = i  # 원하는 비트 폭 (4, 8, 16, 32 중 선택)
    quantized_model = QuantizedModel(original_model, num_bits)
    quantized_model.eval()
    quantized_model.to('cpu')

    import torch_pruning as tp
    from ptflops import get_model_complexity_info
    correct_val = 0
    total_val = 0

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = quantized_model(inputs.to('cpu'))
            _, predicted_val = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val.to('cpu') == labels.to('cpu')).sum().item()

        val_acc = correct_val / total_val  # Validation accuracy for this epoch
        end_time = time.time()
        print(val_acc)
        acc.append(val_acc)
        total_inference_time = 0.0
        start_time = time.time()
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to('cpu')
                    outputs = quantized_model(inputs)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
        end_time = time.time()
        runtime = total_inference_time / 100
        print(f'Prediction Runtime: {runtime:.2f} seconds')
        predic_time.append(runtime)
        total_params = sum(p.numel() for p in quantized_model.parameters())
        print(f'{total_params:,} total parameters.')
        example_inputs = torch.randn(1,3, 48, 48)
        prune_macs, prune_nparams = tp.utils.count_ops_and_params(quantized_model.to('cpu'), example_inputs.to('cpu'))
        print(f'\tMACs: {prune_macs / 1e6} M')
        Macs.append(prune_macs / 1e6)
        print(f'\tParams: {prune_nparams / 1e6} M')

        # 저장 경로 설정
        quantized_model_path = "quantized_model.pth"

        # 양자화된 모델 저장
        quantized_model.save_model(quantized_model_path)

        # 모델 파일 크기 확인
        file_size_mb = os.path.getsize(quantized_model_path) / (1024 ** 2)  # 바이트 -> MB 변환
        print(f"Quantized Model File Size: {file_size_mb:.2f} MB")
        size.append(file_size_mb)
print(acc)
print(predic_time)
print(Macs)
print(size)
