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
        self.quant = QuantStub()  # Quantization module
        self.dequant = DeQuantStub()  # Dequantization module

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
        x = self.quant(x)  # Quantize input
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
        x = self.dequant(x)  # Dequantize output
        return x

# Model parameters
num_classes = 5
input_shape = (48, 48, 3)

# Instantiate the model
model = EmotionRecognitionModel(num_classes=num_classes)

# Load model
model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location = torch.device('cuda')))
'''
model.to('cpu')
torch.backends.quantized.engine = 'qnnpack'

backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)

model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model,  # Pass the pre-trained model here
    {torch.nn.Linear},  # Specify layers to quantize (e.g., Linear layers)
    dtype=torch.qint8  # Use 8-bit integers for weights
)'''

emotion_mapping = {
    0: "good",
    1: "bad",
    2: "bad",
    3: "acceptable",
    4: "acceptable"
}

model.eval()

cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM),format=NV12,width=640,height=480,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise IOError("Cannot Open Camera")

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tts_bad_path = "bad_alert.wav"
last_sound_time = time.time()
model.to('cuda')
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

        prediction = model(finalimage.to('cuda'))
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

