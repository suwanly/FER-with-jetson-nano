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
# %%
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings

import torch_pruning as tp
import time
from torch.quantization import QuantStub, DeQuantStub

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import cv2
from ptflops import get_model_complexity_info


# %%
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionRecognitionModel, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

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
        x = self.quant(x)
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
        x = self.dequant(x)
        return x


# %%
model = EmotionRecognitionModel()
model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location=torch.device('cuda')))

# %%

example_inputs = torch.randn(1, 3, 48, 48)

# %%
# pruning 적용 함수
import torch.nn.utils.prune as prune

conv_prune = 0.8
fc_prune = 0.8
model = EmotionRecognitionModel()
model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location=torch.device('cuda')))


def apply_pruning(model, conv_prune=conv_prune, fc_prune=fc_prune):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=conv_prune, n=2, dim=0)  # 필터의 20%를 pruning
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=fc_prune, n=2, dim=0)  # FC 레이어의 뉴런 40%를 pruning


# pruning 완료 후 가중치에서 마스크 제거
def finalize_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, 'weight')


apply_pruning(model)
finalize_pruning(model)


# 가중치 값이 0인 인덱스 찾는 함수
def find_zero_weight_indices(model):
    zero_indices = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            # 출력 채널별로 가중치의 절대값 합 계산
            weight_abs_sum = weight.abs().view(weight.shape[0], -1).sum(dim=1)
            # 합이 0인 경우 (완전히 pruning된 필터)
            zero_channels = (weight_abs_sum == 0).nonzero(as_tuple=False).squeeze().tolist()
            zero_indices[name] = zero_channels
        elif isinstance(module, nn.Linear):
            weight = module.weight.data
            # 각 뉴런의 입력 가중치 절대값 합 계산
            weight_abs_sum = weight.abs().sum(dim=1)
            # 합이 0인 경우 (완전히 pruning된 뉴런)
            zero_neurons = (weight_abs_sum == 0).nonzero(as_tuple=False).squeeze().tolist()
            zero_indices[name] = zero_neurons
    return zero_indices


zero_weight_indices = find_zero_weight_indices(model)
# %%
import copy

model = EmotionRecognitionModel()
model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location=torch.device('cuda')))
net = copy.deepcopy(model)
net.to('cuda')
example_inputs = torch.randn(1,3, 48, 48)
example_inputs.to('cuda')

DG = tp.DependencyGraph().build_dependency(net.to('cuda'), example_inputs=example_inputs.to('cuda'))

pruning_idxs = zero_weight_indices['conv1']
pruning_group = DG.get_pruning_group(net.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs)

if DG.check_pruning_group(pruning_group):
    pruning_group.prune()

pruning_idxs = zero_weight_indices['conv2']
pruning_group = DG.get_pruning_group(net.conv2, tp.prune_conv_out_channels, idxs=pruning_idxs)

if DG.check_pruning_group(pruning_group):
    pruning_group.prune()

pruning_idxs = zero_weight_indices['conv3']
pruning_group = DG.get_pruning_group(net.conv3, tp.prune_conv_out_channels, idxs=pruning_idxs)

if DG.check_pruning_group(pruning_group):
    pruning_group.prune()

pruning_idxs = zero_weight_indices['conv4']
pruning_group = DG.get_pruning_group(net.conv4, tp.prune_conv_out_channels, idxs=pruning_idxs)

if DG.check_pruning_group(pruning_group):
    pruning_group.prune()

pruning_idxs = zero_weight_indices['fc1']
pruning_group = DG.get_pruning_group(net.fc1, tp.prune_linear_out_channels, idxs=pruning_idxs)

pruning_idxs = zero_weight_indices['fc2']
pruning_group = DG.get_pruning_group(net.fc2, tp.prune_linear_out_channels, idxs=pruning_idxs)

if DG.check_pruning_group(pruning_group):
    pruning_group.prune()

# 3. prune all grouped layer that is coupled with model.conv1
if DG.check_pruning_group(pruning_group):
    pruning_group.prune()

base_macs, base_nparams = tp.utils.count_ops_and_params(model.to('cuda'), example_inputs.to('cuda'))
print('======================')
print('Before')
print(f'\tMACs: {base_macs / 1e6} M')
print(f'\tParams: {base_nparams / 1e6} M')
prune_macs, prune_nparams = tp.utils.count_ops_and_params(net.to('cuda'), example_inputs.to('cuda'))
print('After')
print(f'\tMACs: {prune_macs / 1e6} M')
print(f'\tParams: {prune_nparams / 1e6} M')
print('======================')
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
net.eval()
net.to('cuda')
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

        prediction = net(finalimage.to('cuda'))
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

