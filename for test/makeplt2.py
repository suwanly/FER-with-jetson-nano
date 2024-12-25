# 필요한 라이브러리 임포트
import warnings
warnings.filterwarnings('ignore')  # 경고 무시

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch_pruning as tp
import torch.nn.utils.prune as prune
import time
import os
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

# ptflops 라이브러리 임포트 (FLOPs 계산을 위해 추가)
from ptflops import get_model_complexity_info

# MPS 지원 여부 확인 및 디바이스 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 모델 정의
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(EmotionRecognitionModel, self).__init__()
        self.quant = nn.Identity()  # 양자화 스텁 대체
        self.dequant = nn.Identity()

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

# 데이터 로딩
val_path = 'images/validation/'  # 검증 데이터 경로
# 클래스 이름에 따라 데이터 로드
x_val = []
y_val = []
for class_idx, class_name in enumerate(['happy', 'sad', 'fear', 'surprise', 'neutral']):
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

# 프루닝 함수 정의
def apply_pruning(model, conv_prune=0.0, fc_prune=0.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if conv_prune > 0:
                # 최대 프루닝 비율 계산 (최소 한 개의 채널은 남겨둠)
                num_channels = module.weight.shape[0]
                max_prune = (num_channels - 1) / num_channels
                prune_amount = min(conv_prune, max_prune)
                prune.ln_structured(module, name='weight', amount=prune_amount, n=2, dim=0)
        elif isinstance(module, nn.Linear):
            if fc_prune > 0:
                num_units = module.weight.shape[0]
                max_prune = (num_units - 1) / num_units
                prune_amount = min(fc_prune, max_prune)
                prune.ln_structured(module, name='weight', amount=prune_amount, n=2, dim=0)

def finalize_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 모듈이 프루닝되었는지 확인
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

# 0 가중치 인덱스 찾는 함수
def find_zero_weight_indices(model):
    zero_indices = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            weight_abs_sum = weight.abs().view(weight.shape[0], -1).sum(dim=1)
            zero_channels = (weight_abs_sum == 0).nonzero(as_tuple=False).squeeze().tolist()
            if isinstance(zero_channels, int):
                zero_channels = [zero_channels]
            zero_indices[name] = zero_channels
        elif isinstance(module, nn.Linear):
            weight = module.weight.data
            weight_abs_sum = weight.abs().sum(dim=1)
            zero_neurons = (weight_abs_sum == 0).nonzero(as_tuple=False).squeeze().tolist()
            if isinstance(zero_neurons, int):
                zero_neurons = [zero_neurons]
            zero_indices[name] = zero_neurons
    return zero_indices

# 모델 사이즈 측정 함수
def get_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    size = os.path.getsize("tmp.pt") / 1e6
    os.remove('tmp.pt')
    return size

# 결과 저장을 위한 리스트 초기화
conv_prune_list = []
inference_time_list = []
accuracy_list = []
model_size_list = []
num_params_list = []
macs_list = []
flops_list = []  # FLOPs 결과를 저장할 리스트 추가

# 프루닝 비율에 따른 루프
for conv_prune in np.arange(0.0, 1.0, 0.1):
    print(f"Pruning Conv layers with ratio: {conv_prune}")
    fc_prune = 0.0  # FC 레이어는 프루닝하지 않음

    # 모델 초기화 및 가중치 로드
    model = EmotionRecognitionModel()
    model.load_state_dict(torch.load('MyModelFaceRecogD3.pth', map_location=device))

    # 프루닝 적용
    apply_pruning(model, conv_prune=conv_prune, fc_prune=fc_prune)
    finalize_pruning(model)

    # 0 가중치 인덱스 찾기
    zero_weight_indices = find_zero_weight_indices(model)

    # torch_pruning을 사용하여 모델 구조 조정
    net = copy.deepcopy(model)
    net.to(device)

    example_inputs = torch.randn(1, 3, 48, 48).to(device)

    DG = tp.DependencyGraph().build_dependency(net, example_inputs=example_inputs)

    # Conv 레이어들에 대해 프루닝 적용
    for conv_layer_name in ['conv1', 'conv2', 'conv3', 'conv4']:
        module = getattr(net, conv_layer_name)
        pruning_idxs = zero_weight_indices.get(conv_layer_name, [])
        if pruning_idxs:
            pruning_group = DG.get_pruning_group(module, tp.prune_conv_out_channels, idxs=pruning_idxs)
            if DG.check_pruning_group(pruning_group):
                pruning_group.prune()

    # 모델 평가
    net.eval()
    correct_val = 0
    total_val = 0

    # 추론 시간을 100회 반복하여 측정
    total_inference_time = 0.0
    for _ in range(100):
        start_time = time.time()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = net(inputs)
        end_time = time.time()
        total_inference_time += (end_time - start_time)

    average_inference_time = total_inference_time / 100

    # 정확도는 한 번만 계산
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted_val = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    val_acc = correct_val / total_val  # 검증 정확도

    model_size = get_model_size(net)

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in net.parameters())

    # MACs 및 FLOPs 계산
    prune_macs, prune_nparams = tp.utils.count_ops_and_params(net.to(device), example_inputs)
    macs_list.append(prune_macs)

    # ptflops를 사용하여 FLOPs 계산
    with torch.cuda.device(0 if torch.cuda.is_available() else -1):
        flops, params = get_model_complexity_info(net, (3, 48, 48), as_strings=False, print_per_layer_stat=False)
    flops_list.append(flops)

    # 결과 저장
    conv_prune_list.append(conv_prune)
    inference_time_list.append(average_inference_time)
    accuracy_list.append(val_acc)
    model_size_list.append(model_size)
    num_params_list.append(total_params)

# 결과 시각화
plt.figure(figsize=(16, 12))

plt.subplot(3, 2, 1)
plt.plot(conv_prune_list, accuracy_list, marker='o')
plt.xlabel('Conv Pruning Ratio')
plt.ylabel('Validation Accuracy')
plt.title('Accuracy vs Conv Pruning Ratio')

plt.subplot(3, 2, 2)
plt.plot(conv_prune_list, inference_time_list, marker='o')
plt.xlabel('Conv Pruning Ratio')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time vs Conv Pruning Ratio')

plt.subplot(3, 2, 3)
plt.plot(conv_prune_list, model_size_list, marker='o')
plt.xlabel('Conv Pruning Ratio')
plt.ylabel('Model Size (MB)')
plt.title('Model Size vs Conv Pruning Ratio')

plt.subplot(3, 2, 4)
plt.plot(conv_prune_list, num_params_list, marker='o')
plt.xlabel('Conv Pruning Ratio')
plt.ylabel('Number of Parameters')
plt.title('Number of Parameters vs Conv Pruning Ratio')

plt.subplot(3, 2, 5)
plt.plot(conv_prune_list, macs_list, marker='o')
plt.xlabel('Conv Pruning Ratio')
plt.ylabel('MACs')
plt.title('MACs vs Conv Pruning Ratio')

plt.subplot(3, 2, 6)
plt.plot(conv_prune_list, flops_list, marker='o')
plt.xlabel('Conv Pruning Ratio')
plt.ylabel('FLOPs')
plt.title('FLOPs vs Conv Pruning Ratio')

plt.tight_layout()
plt.show()
