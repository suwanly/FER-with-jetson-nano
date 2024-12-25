import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
# Define the model architecture
from torch.quantization import QuantStub, DeQuantStub
from gtts import gTTS
import time
import os
from playsound import playsound
import simpleaudio as sa


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

# Model parameters
num_classes = 5
input_shape = (48, 48, 3)

# Instantiate the model
model = EmotionRecognitionModel(num_classes=num_classes)

# Load model
model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location = torch.device('cuda')))
original_model = EmotionRecognitionModel()
# %%
original_model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location=torch.device('cpu')))

num_bits = 4  # 원하는 비트 폭 (4, 8, 16, 32 중 선택)
quantized_model = QuantizedModel(original_model, num_bits)
quantized_model.eval()
quantized_model.to('cpu')

emotion_mapping = {
    0: "good",
    1: "bad",
    2: "bad",
    3: "acceptable",
    4: "acceptable"
}

quantized_model.eval()

cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM),format=NV12,width=640,height=480,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise IOError("Cannot Open Camera")

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tts_bad_path = "bad_alert.wav"
last_sound_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        largest_face = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_face
        roi_color = frame[y:y + h, x:x + w]

        face_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        finalimage = cv2.resize(face_roi, (48, 48))
        finalimage = torch.tensor(finalimage, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        prediction = quantized_model(finalimage.to('cpu'))
        label_idx = torch.argmax(prediction).item()
        status = emotion_mapping[label_idx]

        if status == "bad":
            current_time = time.time()
            # 마지막 소리 출력 시간으로부터 10초가 지났는지 확인
            if current_time - last_sound_time > 10:
               	mp3 = sa.WaveObject.from_wave_file("bad_alert.wav")
                play_obj = mp3.play()
                last_sound_time = current_time  # 현재 시간을 마지막 출력 시간으로 갱신
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Face Emotion Recognition', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

