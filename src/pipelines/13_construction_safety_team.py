# 코드 요약: RF-DETR 분할, EfficientNet 분류, ByteTrack과 상태 판정을 통합한 최종 파이프라인입니다.
# 원작성자(노션): 김동혁
# 노션 완료 순서: 13 - RF-DETR+EfficientNet 고도화
# 데이터, 모델 가중치와 생성 결과는 저장소에 포함하지 않습니다.

# %% [markdown]
# # 공사장 안전고리 판별 시스템 v2 (Google Colab)
#
# EfficientNet-B0 체결/미체결 분류 + RF-DETR Segmentation + ByteTrack +
# 작업자별 시간 상태 머신을 결합한 전체 코드입니다.
#
# 실행 전 아래 CONFIG의 경로와 RF-DETR 클래스 이름을 반드시 확인하세요.
# 이 파일은 VS Code의 `# %%` 셀 단위 실행용입니다. 전체 파일을 한 번에 실행하면
# 학습, YouTube 다운로드, 영상 추론까지 연속 수행되므로 팀 공유 시 Notebook 사용을 권장합니다.
#
# ## 팀원용 빠른 실행 순서
#
# 1. 패키지 설치 → import → 프로젝트 경로/CONFIG 셀을 실행합니다.
# 2. 새로 학습하려면 1~8번을 순서대로 실행합니다.
# 3. 저장된 EfficientNet 가중치를 사용하면 6번 학습은 건너뜁니다.
# 4. 영상 추론은 10번(모델 로드) → 11번(상태 로직) → 12번(영상 처리) 순서입니다.
# 5. 로컬에서는 환경변수 `CONSTRUCTION_SAFETY_PROJECT_ROOT` 또는
#    `PROJECT_ROOT` 값을 실제 프로젝트 폴더로 바꾸세요.

# %%
# Colab 패키지 설치 (설치 후 런타임 재시작이 요구되면 재시작 후 다음 셀부터 실행)
# 로컬 터미널에서 최초 1회 실행:
# python -m pip install "rfdetr==1.8.0" "supervision>=0.29.0" scikit-learn seaborn opencv-python-headless yt-dlp

# %%
import os
import csv
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
    precision_recall_curve,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# %%
# ============================================================
# 실행 환경과 프로젝트 루트 설정
# ============================================================
IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount("/content/drive")
except ImportError:
    print("Local/Jupyter 환경: Google Drive mount를 건너뜁니다.")

# Colab은 Drive 경로, 로컬은 환경변수 또는 현재 작업 폴더를 기본값으로 사용합니다.
# 로컬 PowerShell 예시:
#   $env:CONSTRUCTION_SAFETY_PROJECT_ROOT="C:\my_project\construction_safety"
if IN_COLAB:
    DEFAULT_PROJECT_ROOT = Path("/content/drive/MyDrive/0_ASAC_11기_DL_1조")
else:
    DEFAULT_PROJECT_ROOT = Path(
        os.environ.get("CONSTRUCTION_SAFETY_PROJECT_ROOT", Path.cwd())
    )

# 환경변수를 쓰지 않는 팀원은 아래 값을 Path("...")로 직접 바꿔도 됩니다.
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pipeline"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Execution environment:", "Google Colab" if IN_COLAB else "Local/Jupyter")
print("Project root:", PROJECT_ROOT)
print("Output root:", OUTPUT_ROOT)


# %%
# ============================================================
# 0. 전체 설정: 이 셀의 경로만 먼저 수정하세요.
# ============================================================
CONFIG = {
    # 데이터 폴더: 하위 경로 이름에 connected / unconnected가 있어야 합니다.
    "data_root": str(PROJECT_ROOT / "data" / "classification"),
    "classifier_checkpoint": str(
        PROJECT_ROOT
        / "models"
        / "classification"
        / "efficientnet"
        / "efficientnet_hook_classifier_best.pth"
    ),
    "metrics_csv": str(OUTPUT_ROOT / "efficientnet_training_metrics.csv"),

    # 최초 작성자가 제공한 이전 EfficientNet 가중치: 비교용이며 새 학습에는 필수가 아닙니다.
    # 실제 파일명이 다르면 수정하세요.
    "baseline_classifier_checkpoint": str(
        PROJECT_ROOT
        / "models"
        / "classification"
        / "efficientnet"
        / "efficientnet_hook_classifier_legacy.pth"
    ),

    # RF-DETR 및 영상
    "rfdetr_checkpoint": str(
        PROJECT_ROOT
        / "models"
        / "detection"
        / "rfdetr"
        / "checkpoint_best_ema_v2.pth"
    ),
    "input_video": str(PROJECT_ROOT / "samples" / "videos" / "input_video.mp4"),
    "output_video": str(OUTPUT_ROOT / "pipeline_output_v2.mp4"),
    "event_log_csv": str(OUTPUT_ROOT / "safety_events.csv"),

    # 재현성 및 학습
    "seed": 42,
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 2,
    "epochs": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    # Train의 connected/unconnected 비율로 loss weight를 자동 계산합니다.
    # 두 클래스에 동일한 online augmentation을 적용하고 실제 클래스 비율로 loss를 보정합니다.
    # 과도한 DANGER 편향을 막기 위해 Weighted sampler는 사용하지 않습니다.
    "auto_class_weight": True,
    "fallback_danger_loss_weight": 2.0,
    "use_weighted_sampler": False,
    "min_danger_recall": 0.95,
    "max_false_alarm_rate": 0.10,
    "early_stopping_patience": 5,

    # 데이터 분할
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,

    # RF-DETR는 낮은 base threshold로 후보를 받은 뒤 클래스별로 필터링합니다.
    "detector_base_threshold": 0.20,
    "class_thresholds": {
        "worker": 0.40,
        "harness": 0.35,
        "hook": 0.30,
        "lanyard": 0.25,
        "lifeline": 0.30,
    },

    # 안전 판정: Validation으로 선택된 threshold를 기본 사용합니다.
    "use_checkpoint_threshold": True,
    "fallback_connected_enter_threshold": 0.85,
    "connected_exit_gap": 0.10,  # enter 0.85라면 exit 0.75
    "ema_alpha": 0.35,

    # 초 단위 상태 조건
    "safe_confirm_seconds": 1.50,
    "danger_confirm_seconds": 0.30,
    "hook_missing_warning_seconds": 1.00,
    "hook_missing_danger_seconds": 3.00,
    "worker_track_ttl_seconds": 3.00,
    "alert_cooldown_seconds": 10.0,

    # 후크 crop과 작업자 매칭
    "hook_crop_margin_ratio": 0.50,
    # 고리는 lanyard 끝에 있어 작업자 bbox 밖으로 멀어질 수 있습니다.
    "worker_box_expand_ratio": 0.50,
    "max_hook_worker_distance_ratio": 1.50,

    # Harness는 현재 보조 증거로 사용. True이면 장시간 미탐 시 WARNING 표시
    "use_harness_as_support": True,
    "harness_missing_warning_seconds": 2.0,
}

assert abs(CONFIG["train_ratio"] + CONFIG["val_ratio"] + CONFIG["test_ratio"] - 1.0) < 1e-9


# %%
# ============================================================
# 1. 재현성 설정 및 디바이스 확인
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type != "cuda":
    print("WARNING: Colab 런타임에서 GPU를 선택하는 것을 권장합니다.")


# %%
# ============================================================
# 2. 데이터 탐색 및 클래스별 8:1:1 분할
# 라벨 0 = connected(SAFE), 라벨 1 = unconnected(DANGER)
# ============================================================
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(root_dir: str):
    """하위 폴더를 재귀 탐색해 connected/unconnected 이미지 경로를 수집합니다.

    폴더 이름을 라벨로 사용하며 JSON annotation 파일은 이미지 확장자가 아니므로
    자동 제외됩니다. 'unconnected' 안에 'connected' 문자열이 포함되므로 반드시
    unconnected를 먼저 검사합니다.
    """
    connected, unconnected = [], []
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {root_dir}")

    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        parent_text = str(path.parent).lower()
        # unconnected를 먼저 검사해야 connected 부분 문자열과 혼동하지 않습니다.
        if "unconnected" in parent_text:
            unconnected.append(str(path))
        elif "connected" in parent_text:
            connected.append(str(path))

    if not connected or not unconnected:
        raise ValueError(
            "connected/unconnected 데이터가 모두 필요합니다. "
            f"connected={len(connected)}, unconnected={len(unconnected)}"
        )
    return connected, unconnected


def split_one_class(paths, train_ratio, val_ratio, seed):
    """한 클래스의 이미지 목록을 재현 가능한 Train/Val/Test 순서로 분할합니다."""
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        paths[:n_train],
        paths[n_train:n_train + n_val],
        paths[n_train + n_val:],
    )


connected_all, unconnected_all = discover_images(CONFIG["data_root"])
c_train, c_val, c_test = split_one_class(
    connected_all, CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["seed"]
)
u_train, u_val, u_test = split_one_class(
    unconnected_all, CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["seed"] + 1
)

train_samples = [(p, 0) for p in c_train] + [(p, 1) for p in u_train]
val_samples = [(p, 0) for p in c_val] + [(p, 1) for p in u_val]
test_samples = [(p, 0) for p in c_test] + [(p, 1) for p in u_test]
random.Random(CONFIG["seed"]).shuffle(train_samples)

split_summary = pd.DataFrame({
    "split": ["train", "validation", "test", "total"],
    "connected": [len(c_train), len(c_val), len(c_test), len(connected_all)],
    "unconnected": [len(u_train), len(u_val), len(u_test), len(unconnected_all)],
})
split_summary["total"] = split_summary["connected"] + split_summary["unconnected"]
display(split_summary)
print("주의: 연속 영상 프레임이라면 영상/촬영 세션 단위로 분할해야 데이터 누수를 피할 수 있습니다.")


# %%
# ============================================================
# 3. Dataset, augmentation, DataLoader
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 후크 crop의 체결 부위를 잘라내지 않는 공통 기본 증강을 두 클래스에 동일 적용합니다.
# 클래스별로 서로 다른 증강을 적용하면 모델이 증강 흔적을 라벨로 학습할 수 있습니다.
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class HookDataset(Dataset):
    """이미지 경로와 정수 라벨을 읽어 PyTorch 학습 tensor로 반환하는 Dataset입니다.

    반환값에 path도 포함해 False SAFE 같은 오분류 원본을 추적할 수 있습니다.
    """
    def __init__(self, samples, transform=None, class_transforms=None):
        self.samples = samples
        self.transform = transform
        self.class_transforms = class_transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"이미지 로드 실패: {path}") from exc
        if self.class_transforms is not None:
            image = self.class_transforms[int(label)](image)
        elif self.transform is not None:
            image = self.transform(image)
        return image, int(label), path


train_dataset = HookDataset(train_samples, transform=train_transform)
val_dataset = HookDataset(val_samples, eval_transform)
test_dataset = HookDataset(test_samples, eval_transform)

sampler = None
shuffle = True
if CONFIG["use_weighted_sampler"]:
    labels = np.array([label for _, label in train_samples])
    counts = np.bincount(labels, minlength=2)
    sample_weights = np.array([1.0 / counts[label] for label in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    shuffle = False

loader_kwargs = {
    "batch_size": CONFIG["batch_size"],
    "num_workers": CONFIG["num_workers"],
    "pin_memory": DEVICE.type == "cuda",
}
train_loader = DataLoader(train_dataset, shuffle=shuffle, sampler=sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)


# %%
# ============================================================
# 4. EfficientNet-B0: Avg + Max pooling, MLP classifier
# ============================================================
class CustomPooling(nn.Module):
    """전체 평균 특징과 가장 강한 국소 특징을 결합하는 pooling입니다."""
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


def build_classifier(pretrained: bool):
    """Custom Pooling과 2-class MLP head를 가진 EfficientNet-B0를 생성합니다.

    pretrained=True이면 ImageNet-1K 사전학습 가중치로 시작하고,
    False이면 저장된 우리 체크포인트를 불러올 빈 구조만 만듭니다.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.avgpool = CustomPooling()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.30, inplace=False),
        nn.Linear(1280 * 2, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=0.30, inplace=False),
        nn.Linear(512, 2),
    )
    return model


classifier = build_classifier(pretrained=True).to(DEVICE)

# 실제 Train 클래스 비율로 미체결(Danger) loss weight 자동 계산
train_connected_count = sum(label == 0 for _, label in train_samples)
train_unconnected_count = sum(label == 1 for _, label in train_samples)

if train_unconnected_count == 0:
    raise ValueError("Train에 unconnected 이미지가 없습니다.")

auto_class_weight = CONFIG.get("auto_class_weight", True)
fallback_danger_loss_weight = CONFIG.get("fallback_danger_loss_weight", 2.0)

if auto_class_weight:
    danger_loss_weight = train_connected_count / train_unconnected_count
else:
    danger_loss_weight = fallback_danger_loss_weight

class_weights = torch.tensor(
    [1.0, danger_loss_weight],
    dtype=torch.float32,
    device=DEVICE,
)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f"Train connected: {train_connected_count}")
print(f"Train unconnected: {train_unconnected_count}")
print(f"Connected loss weight: {class_weights[0].item():.4f}")
print(f"Danger loss weight: {class_weights[1].item():.4f}")
if CONFIG.get("use_weighted_sampler", False) and auto_class_weight:
    print(
        "WARNING: WeightedRandomSampler와 자동 class weight가 동시에 활성화되어 "
        "미체결 클래스가 과도하게 강화될 수 있습니다. 하나만 사용하는 것을 권장합니다."
    )
optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
print(classifier.classifier)


# %%
# ============================================================
# 5. 평가 함수와 connected threshold 탐색
# connected 확률 >= threshold일 때만 SAFE(0), 그 외는 DANGER(1)
# ============================================================
def binary_metrics(y_true, y_pred):
    """Unconnected(라벨 1)을 Positive로 두고 안전 핵심 지표를 계산합니다."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    false_safe_count = int(fn)  # 실제 danger(1)를 safe(0)로 예측
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "accuracy": accuracy,
        "danger_precision": float(precision[0]),
        "danger_recall": float(recall[0]),
        "danger_f1": float(f1[0]),
        "false_safe_count": false_safe_count,
        "false_alarm_rate": false_alarm_rate,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def predictions_from_connected_probability(prob_connected, threshold):
    return np.where(np.asarray(prob_connected) >= threshold, 0, 1)


def choose_connected_threshold(
    y_true,
    prob_connected,
    min_danger_recall,
    max_false_alarm_rate=1.0,
):
    """Validation에서 0.50~0.99 threshold를 비교해 운영 경계값을 선택합니다.

    우선 Recall 최소값과 오경보율 상한을 모두 만족시키고, 그 후보 중
    F1/Precision이 높은 값을 고릅니다. Test 데이터는 선택에 사용하지 않습니다.
    """
    rows = []
    for threshold in np.arange(0.50, 0.991, 0.01):
        pred = predictions_from_connected_probability(prob_connected, threshold)
        metrics = binary_metrics(y_true, pred)
        rows.append({"threshold": round(float(threshold), 2), **metrics})

    table = pd.DataFrame(rows)
    eligible = table[
        (table["danger_recall"] >= min_danger_recall)
        & (table["false_alarm_rate"] <= max_false_alarm_rate)
    ]
    if len(eligible):
        # 안전 Recall 조건을 만족하는 후보 중 F1, Precision, 낮은 오경보 순
        best = eligible.sort_values(
            ["danger_f1", "danger_precision", "false_alarm_rate"],
            ascending=[False, False, True],
        ).iloc[0]
        target_met = True
    else:
        # 두 안전 조건을 동시에 만족하지 못하면 Recall 100%만 노리고 거의 모든
        # 이미지를 DANGER로 보내지 않도록, 오경보 상한 내에서 F1이 가장 높은 값을 선택합니다.
        alarm_limited = table[table["false_alarm_rate"] <= max_false_alarm_rate]
        candidates = alarm_limited if len(alarm_limited) else table
        best = candidates.sort_values(
            ["danger_f1", "danger_recall", "danger_precision"],
            ascending=[False, False, False],
        ).iloc[0]
        target_met = False
    return float(best["threshold"]), best.to_dict(), table, target_met


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn):
    """평가 데이터의 loss, 정답, Connected 확률, 원본 경로를 한 번에 수집합니다."""
    model.eval()
    total_loss = 0.0
    y_true, prob_connected, paths = [], [], []
    for images, labels, batch_paths in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        probs = F.softmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        prob_connected.extend(probs[:, 0].cpu().numpy().tolist())
        paths.extend(batch_paths)
    return total_loss / len(loader.dataset), np.asarray(y_true), np.asarray(prob_connected), paths


# %%
# ============================================================
# 6. 학습: 최소 Danger Recall을 만족한 후보 중 Danger F1 우선 저장
# ============================================================
history = []
best_score = None
epochs_without_improvement = 0
training_start_time = time.time()
completed_epoch_times = []

for epoch in range(1, CONFIG["epochs"] + 1):
    epoch_start_time = time.time()
    classifier.train()
    running_loss = 0.0
    train_correct = 0
    train_count = 0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d}/{CONFIG['epochs']} Train",
        unit="batch",
        leave=True,
    )
    for batch_index, (images, labels, _) in enumerate(train_progress, start=1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_count += labels.size(0)

        train_progress.set_postfix({
            "loss": f"{running_loss / train_count:.4f}",
            "acc": f"{train_correct / train_count:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
        })

    train_loss = running_loss / train_count
    train_acc = train_correct / train_count
    val_loss, val_y, val_prob_c, _ = evaluate_loader(classifier, val_loader, criterion)
    scheduler.step(val_loss)

    threshold, val_metrics, threshold_table, target_met = choose_connected_threshold(
        val_y,
        val_prob_c,
        CONFIG["min_danger_recall"],
        CONFIG.get("max_false_alarm_rate", 0.10),
    )

    epoch_seconds = time.time() - epoch_start_time
    completed_epoch_times.append(epoch_seconds)
    average_epoch_seconds = sum(completed_epoch_times) / len(completed_epoch_times)
    remaining_epochs = CONFIG["epochs"] - epoch
    estimated_remaining_seconds = average_epoch_seconds * remaining_epochs

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "connected_threshold": threshold,
        "target_recall_met": target_met,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "epoch_seconds": epoch_seconds,
        **{k: val_metrics[k] for k in [
            "accuracy", "danger_precision", "danger_recall", "danger_f1",
            "false_safe_count", "false_alarm_rate"
        ]},
    }
    history.append(row)
    pd.DataFrame(history).to_csv(CONFIG["metrics_csv"], index=False)

    # 목표 Recall 만족 여부가 최우선. 그 안에서는 F1/Precision, val loss 순.
    score = (
        int(target_met),
        val_metrics["danger_f1"],
        val_metrics["danger_precision"] if target_met else val_metrics["danger_recall"],
        -val_loss,
    )

    print(
        f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
        f"thr={threshold:.2f} danger_recall={val_metrics['danger_recall']:.3f} "
        f"precision={val_metrics['danger_precision']:.3f} "
        f"F1={val_metrics['danger_f1']:.3f} "
        f"false_SAFE={val_metrics['false_safe_count']} | "
        f"epoch_time={epoch_seconds/60:.1f}m "
        f"estimated_remaining={estimated_remaining_seconds/60:.1f}m"
    )

    if best_score is None or score > best_score:
        best_score = score
        epochs_without_improvement = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "connected_threshold": threshold,
            "val_metrics": val_metrics,
            "config": CONFIG,
            "class_names": ["connected", "unconnected"],
        }, CONFIG["classifier_checkpoint"])
        print("  -> Best checkpoint saved")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
        print("Early stopping")
        break

print(f"Total training time: {(time.time() - training_start_time)/60:.1f} minutes")
display(pd.DataFrame(history))


# %%
# ============================================================
# 7. 학습 곡선
# ============================================================
history_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], label="validation")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(history_df["epoch"], history_df["danger_recall"], label="Danger recall")
axes[1].plot(history_df["epoch"], history_df["danger_precision"], label="Danger precision")
axes[1].plot(history_df["epoch"], history_df["danger_f1"], label="Danger F1")
axes[1].axhline(CONFIG["min_danger_recall"], color="red", linestyle="--")
axes[1].set_ylim(0, 1.05)
axes[1].legend()
axes[2].plot(history_df["epoch"], history_df["connected_threshold"])
axes[2].set_title("Selected connected threshold")
plt.show()


# %%
# ============================================================
# 8. Test 최종 평가: 이 결과를 보고 threshold를 다시 조정하지 마세요.
# ============================================================
checkpoint = torch.load(CONFIG["classifier_checkpoint"], map_location=DEVICE)
classifier = build_classifier(pretrained=False).to(DEVICE)
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.eval()

SELECTED_CONNECTED_THRESHOLD = float(checkpoint["connected_threshold"])
test_loss, test_y, test_prob_c, test_paths = evaluate_loader(classifier, test_loader, criterion)
test_pred = predictions_from_connected_probability(test_prob_c, SELECTED_CONNECTED_THRESHOLD)
test_metrics = binary_metrics(test_y, test_pred)

print("Selected connected threshold:", SELECTED_CONNECTED_THRESHOLD)
print("Test loss:", round(test_loss, 4))
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
print(classification_report(
    test_y, test_pred,
    labels=[0, 1],
    target_names=["connected", "unconnected"],
    digits=4,
    zero_division=0,
))

cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred SAFE", "Pred DANGER"],
            yticklabels=["True connected", "True unconnected"])
plt.title("Test confusion matrix")
plt.show()

error_df = pd.DataFrame({
    "path": test_paths,
    "true_label": test_y,
    "pred_label": test_pred,
    "prob_connected": test_prob_c,
})
false_safe_df = error_df[(error_df.true_label == 1) & (error_df.pred_label == 0)]
print("가장 위험한 False SAFE 샘플 수:", len(false_safe_df))
display(false_safe_df.sort_values("prob_connected", ascending=False).head(20))


# %%
# ============================================================
# 9. 단일 이미지 추론 함수
# ============================================================
@torch.no_grad()
def predict_hook_image(image_path, threshold=None):
    threshold = SELECTED_CONNECTED_THRESHOLD if threshold is None else float(threshold)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    probability = F.softmax(classifier(tensor), dim=1)[0]
    prob_connected = float(probability[0].item())
    status = "SAFE" if prob_connected >= threshold else "DANGER"
    result = {
        "status": status,
        "connected_probability": prob_connected,
        "unconnected_probability": 1.0 - prob_connected,
        "connected_threshold": threshold,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

# 예시:
# predict_hook_image("/content/test_hook.jpg")


# %% [markdown]
# ## 영상 파이프라인
#
# 아래부터는 학습이 끝난 EfficientNet 체크포인트와 RF-DETR 체크포인트를 사용합니다.
# 작업자별 상태를 유지하므로 후크가 잠시 사라져도 WARNING/DANGER를 판단합니다.

# %%
# ============================================================
# 선택 실행: YouTube 테스트 영상을 구간별로 Drive에 저장
# 저작권 및 이용 권한이 있는 영상의 연구/테스트 용도로만 사용하세요.
# 두 번째 영상은 시간 범위가 없어 전체 영상(None, None)으로 설정했습니다.
# ============================================================
import subprocess

YOUTUBE_CLIP_ROOT = PROJECT_ROOT / "youtube_test_clips"
YOUTUBE_CLIP_ROOT.mkdir(parents=True, exist_ok=True)

YOUTUBE_CLIPS = [
    {
        "url": "https://www.youtube.com/watch?v=zoSQKeOpSF0",
        "start": "00:20:30",
        "end": None,  # 영상 끝까지
        "name": "youtube_01_from_20m30s",
    },
    {
        "url": "https://www.youtube.com/watch?v=18AdC40I2oU",
        "start": None,
        "end": None,  # 전체 영상. 필요한 구간이 있으면 시간을 입력하세요.
        "name": "youtube_02_full",
    },
    {
        "url": "https://www.youtube.com/watch?v=l_3U5YqG8PQ",
        "start": "00:08:15",
        "end": "00:14:00",
        "name": "youtube_03_08m15s_to_14m00s",
    },
    {
        "url": "https://www.youtube.com/watch?v=l_3U5YqG8PQ",
        "start": "00:19:30",
        "end": "00:20:00",
        "name": "youtube_03_19m30s_to_20m00s",
    },
    {
        "url": "https://www.youtube.com/watch?v=lGDjYTwh3MQ",
        "start": "00:00:00",
        "end": "00:01:47",
        "name": "youtube_04_00m00s_to_01m47s",
    },
]


def download_youtube_clip(item, overwrite=False):
    """yt-dlp로 영상 전체 또는 지정 구간을 720p 이하로 내려받습니다."""
    output_template = str(YOUTUBE_CLIP_ROOT / f"{item['name']}.%(ext)s")
    existing_mp4 = YOUTUBE_CLIP_ROOT / f"{item['name']}.mp4"
    if existing_mp4.exists() and not overwrite:
        print("SKIP (already exists):", existing_mp4)
        return existing_mp4

    command = [
        "yt-dlp",
        "--no-playlist",
        "--newline",
        "--merge-output-format", "mp4",
        "--force-keyframes-at-cuts",
        "-f", "bv*[height<=720]+ba/b[height<=720]",
        "-o", output_template,
    ]

    start, end = item["start"], item["end"]
    if start is not None or end is not None:
        section_start = start or "00:00:00"
        section_end = end or "inf"
        command.extend(["--download-sections", f"*{section_start}-{section_end}"])

    if overwrite:
        command.append("--force-overwrites")

    command.append(item["url"])
    print("\nDownloading:", item["name"])
    print("Range:", start or "START", "~", end or "END")
    subprocess.run(command, check=True)

    candidates = sorted(YOUTUBE_CLIP_ROOT.glob(f"{item['name']}.*"))
    if not candidates:
        raise FileNotFoundError(f"다운로드 결과를 찾지 못했습니다: {item['name']}")
    print("Saved:", candidates[0])
    return candidates[0]


# 전체 항목 다운로드. 이미 존재하는 파일은 건너뜁니다.
# 두 번째 전체 영상이 너무 크면 YOUTUBE_CLIPS에서 시간을 먼저 지정하세요.
downloaded_test_videos = [
    download_youtube_clip(item, overwrite=False)
    for item in YOUTUBE_CLIPS
]

print("\nDownloaded files:")
for video_path in downloaded_test_videos:
    print(video_path)

# %%
# ============================================================
# 10. RF-DETR, 추적기 로드 및 클래스 확인
# ============================================================
import supervision as sv
from rfdetr import RFDETRSegLarge

if not os.path.exists(CONFIG["rfdetr_checkpoint"]):
    raise FileNotFoundError(f"RF-DETR 체크포인트가 없습니다: {CONFIG['rfdetr_checkpoint']}")
if not os.path.exists(CONFIG["classifier_checkpoint"]):
    raise FileNotFoundError(f"EfficientNet 체크포인트가 없습니다: {CONFIG['classifier_checkpoint']}")

detector = RFDETRSegLarge(pretrain_weights=CONFIG["rfdetr_checkpoint"])
print("RF-DETR class_names (0-based):", detector.class_names)
print("반드시 worker, harness, hook, lanyard, lifeline 이름이 실제 출력과 일치하는지 확인하세요.")

checkpoint = torch.load(CONFIG["classifier_checkpoint"], map_location=DEVICE)
classifier = build_classifier(pretrained=False).to(DEVICE)
classifier.load_state_dict(checkpoint["model_state_dict"])
classifier.eval()

if CONFIG["use_checkpoint_threshold"]:
    CONNECTED_ENTER_THRESHOLD = float(checkpoint["connected_threshold"])
else:
    CONNECTED_ENTER_THRESHOLD = CONFIG["fallback_connected_enter_threshold"]
CONNECTED_EXIT_THRESHOLD = max(0.0, CONNECTED_ENTER_THRESHOLD - CONFIG["connected_exit_gap"])
print("Connected enter threshold:", CONNECTED_ENTER_THRESHOLD)
print("Connected exit threshold:", CONNECTED_EXIT_THRESHOLD)


# %%
# ============================================================
# 11. 영상 유틸리티와 작업자 상태 머신
# ============================================================
def normalize_class_name(name):
    return str(name).strip().lower().replace(" ", "_")


def detection_class_name(detections, index, detector_class_names):
    """RF-DETR 결과에서 숫자 ID보다 안전한 문자열 클래스 이름을 반환합니다."""
    if hasattr(detections, "data") and "class_name" in detections.data:
        return normalize_class_name(detections.data["class_name"][index])
    class_id = int(detections.class_id[index])
    if 0 <= class_id < len(detector_class_names):
        return normalize_class_name(detector_class_names[class_id])
    return f"class_{class_id}"


def filter_detections_by_class_threshold(detections, detector_class_names):
    """작은 Hook/Lanyard와 Worker에 서로 다른 최소 confidence를 적용합니다."""
    keep = []
    for i in range(len(detections)):
        name = detection_class_name(detections, i, detector_class_names)
        minimum = CONFIG["class_thresholds"].get(name, CONFIG["detector_base_threshold"])
        keep.append(float(detections.confidence[i]) >= minimum)
    return detections[np.asarray(keep, dtype=bool)]


def box_center(box):
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_expanded_box(point, box, expand_ratio):
    x, y = point
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1
    return (
        x1 - w * expand_ratio <= x <= x2 + w * expand_ratio
        and y1 - h * expand_ratio <= y <= y2 + h * expand_ratio
    )


def normalized_center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    x1, y1, x2, y2 = map(float, box_b)
    diagonal = max(math.hypot(x2 - x1, y2 - y1), 1.0)
    return math.hypot(ax - bx, ay - by) / diagonal


def greedy_match_objects_to_workers(workers, objects):
    """각 장비를 최대 한 작업자에게 거리 기반으로 연결합니다.

    점수가 낮을수록 가깝습니다. 한 작업자/장비가 중복 배정되지 않도록 Greedy하게
    연결합니다. 다중 작업자 환경에서는 관계 추론보다 단순한 근사 방식입니다.
    """
    candidates = []
    for wi, worker in enumerate(workers):
        for oi, obj in enumerate(objects):
            center = box_center(obj["bbox"])
            inside = point_in_expanded_box(
                center, worker["bbox"], CONFIG["worker_box_expand_ratio"]
            )
            distance = normalized_center_distance(obj["bbox"], worker["bbox"])
            if inside or distance <= CONFIG["max_hook_worker_distance_ratio"]:
                score = distance - (0.25 if inside else 0.0)
                candidates.append((score, wi, oi))

    matches = {}
    used_workers, used_objects = set(), set()
    for _, wi, oi in sorted(candidates):
        if wi in used_workers or oi in used_objects:
            continue
        matches[workers[wi]["id"]] = objects[oi]
        used_workers.add(wi)
        used_objects.add(oi)
    return matches


@torch.no_grad()
def classify_hook_crop(frame_rgb, hook_box):
    """Hook 탐지 박스에 여백을 추가해 Crop하고 Connected 확률을 반환합니다."""
    x1, y1, x2, y2 = map(int, hook_box)
    hook_w, hook_h = max(1, x2 - x1), max(1, y2 - y1)
    margin = int(max(hook_w, hook_h) * CONFIG["hook_crop_margin_ratio"])
    height, width = frame_rgb.shape[:2]
    cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
    cx2, cy2 = min(width, x2 + margin), min(height, y2 + margin)
    crop = frame_rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    tensor = eval_transform(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
    probs = F.softmax(classifier(tensor), dim=1)[0]
    return float(probs[0].item())


@dataclass
class WorkerSafetyState:
    """작업자 한 명의 시간축 안전 상태와 최근 탐지 정보를 보관합니다."""
    worker_id: int
    first_seen: float
    last_seen: float
    last_hook_seen: float | None = None
    last_harness_seen: float | None = None
    ema_connected: float | None = None
    safe_candidate_since: float | None = None
    danger_candidate_since: float | None = None
    status: str = "UNKNOWN"
    previous_status: str = "UNKNOWN"
    reason: str = "initializing"
    last_alert_time: float = -1e9

    def update(self, now, connected_probability, hook_seen, harness_seen):
        """현재 프레임 정보를 반영해 UNKNOWN/WARNING/SAFE/DANGER 상태를 갱신합니다."""
        self.previous_status = self.status
        self.last_seen = now

        if harness_seen:
            self.last_harness_seen = now

        if hook_seen and connected_probability is not None:
            self.last_hook_seen = now
            if self.ema_connected is None:
                self.ema_connected = connected_probability
            else:
                alpha = CONFIG["ema_alpha"]
                self.ema_connected = alpha * connected_probability + (1 - alpha) * self.ema_connected

            # SAFE 진입은 높은 threshold + 충분한 지속시간이 필요합니다.
            if self.ema_connected >= CONNECTED_ENTER_THRESHOLD:
                self.danger_candidate_since = None
                if self.safe_candidate_since is None:
                    self.safe_candidate_since = now
                safe_duration = now - self.safe_candidate_since
                if safe_duration >= CONFIG["safe_confirm_seconds"]:
                    self.status = "SAFE"
                    self.reason = "hook connected continuously"
                else:
                    self.status = "PENDING_SAFE"
                    self.reason = f"confirming connection {safe_duration:.1f}s"

            # SAFE 해제는 더 낮은 exit threshold를 사용하여 깜빡임을 줄입니다.
            elif self.ema_connected < CONNECTED_EXIT_THRESHOLD:
                self.safe_candidate_since = None
                if self.danger_candidate_since is None:
                    self.danger_candidate_since = now
                danger_duration = now - self.danger_candidate_since
                if danger_duration >= CONFIG["danger_confirm_seconds"]:
                    self.status = "DANGER"
                    self.reason = "hook classified as unconnected"
                else:
                    self.status = "WARNING"
                    self.reason = "confirming danger"
            else:
                # 히스테리시스 구간에서는 기존 확정 상태를 유지합니다.
                self.safe_candidate_since = None
                self.danger_candidate_since = None
                if self.status not in {"SAFE", "DANGER"}:
                    self.status = "WARNING"
                self.reason = "uncertain connection probability"

        else:
            self.safe_candidate_since = None
            reference = self.last_hook_seen if self.last_hook_seen is not None else self.first_seen
            missing_seconds = now - reference
            if missing_seconds >= CONFIG["hook_missing_danger_seconds"]:
                self.status = "DANGER"
                self.reason = f"hook missing {missing_seconds:.1f}s"
            elif missing_seconds >= CONFIG["hook_missing_warning_seconds"]:
                self.status = "WARNING"
                self.reason = f"hook temporarily missing {missing_seconds:.1f}s"
            elif self.status not in {"SAFE", "DANGER"}:
                self.status = "UNKNOWN"
                self.reason = "waiting for hook detection"

        if CONFIG["use_harness_as_support"]:
            harness_reference = self.last_harness_seen if self.last_harness_seen is not None else self.first_seen
            harness_missing = now - harness_reference
            if harness_missing >= CONFIG["harness_missing_warning_seconds"] and self.status == "SAFE":
                self.status = "WARNING"
                self.reason = f"harness not confirmed for {harness_missing:.1f}s"

        return self.status


STATUS_COLORS = {
    "SAFE": (0, 200, 0),
    "PENDING_SAFE": (0, 220, 220),
    "WARNING": (0, 165, 255),
    "DANGER": (0, 0, 255),
    "UNKNOWN": (160, 160, 160),
}


def draw_worker_status(frame, worker, state):
    x1, y1, x2, y2 = map(int, worker["bbox"])
    color = STATUS_COLORS[state.status]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    probability_text = ""
    if state.ema_connected is not None:
        prefix = "C(last)" if "missing" in state.reason else "C"
        probability_text = f" {prefix}:{state.ema_connected:.2f}"
    label = f"W:{state.worker_id} {state.status}{probability_text}"
    cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, state.reason[:55], (x1, min(frame.shape[0] - 10, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


# %%
# ============================================================
# 12. 최종 영상 처리
# ============================================================
input_path = CONFIG["input_video"]
output_path = CONFIG["output_video"]
if not os.path.exists(input_path):
    raise FileNotFoundError(f"입력 영상이 없습니다: {input_path}")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if width <= 0 or height <= 0 or fps <= 0:
    raise RuntimeError("영상 메타데이터가 올바르지 않습니다.")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not writer.isOpened():
    raise RuntimeError(f"출력 영상을 생성할 수 없습니다: {output_path}")

tracker = sv.ByteTrack(frame_rate=round(fps))
mask_annotator = sv.MaskAnnotator(opacity=0.25)
worker_states = {}
events = []
frame_index = 0
start_wall_time = time.time()

print(f"Video: {width}x{height}, FPS={fps:.2f}, frames={total_frames}")
print("Processing...")

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        now = frame_index / fps
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detections = detector.predict(
            images=frame_rgb,
            threshold=CONFIG["detector_base_threshold"],
            include_source_image=False,
        )
        detections = filter_detections_by_class_threshold(detections, detector.class_names)
        detections = tracker.update_with_detections(detections)

        workers, hooks, harnesses = [], [], []
        for i in range(len(detections)):
            if detections.tracker_id is None:
                continue
            tracker_id = int(detections.tracker_id[i])
            name = detection_class_name(detections, i, detector.class_names)
            item = {
                "id": tracker_id,
                "bbox": detections.xyxy[i].copy(),
                "confidence": float(detections.confidence[i]),
                "name": name,
            }
            if name == "worker":
                workers.append(item)
            elif name == "hook":
                hooks.append(item)
            elif name == "harness":
                harnesses.append(item)

        hook_matches = greedy_match_objects_to_workers(workers, hooks)
        harness_matches = greedy_match_objects_to_workers(workers, harnesses)

        # 마스크는 배경 정보로만 낮은 투명도로 표시합니다.
        annotated = frame_bgr.copy()
        if getattr(detections, "mask", None) is not None and len(detections):
            annotated = mask_annotator.annotate(scene=annotated, detections=detections)

        # RF-DETR가 탐지한 모든 안전장비를 확인할 수 있도록 표시합니다.
        # worker와 매칭된 hook은 아래 안전 판정 단계에서 녹색/빨간색으로 다시 그립니다.
        equipment_colors = {
            "harness": (255, 255, 0),   # cyan
            "hook": (0, 165, 255),      # orange (아직 매칭 전)
            "lanyard": (255, 0, 255),   # magenta
            "lifeline": (255, 120, 0),  # blue
        }
        for i in range(len(detections)):
            name = detection_class_name(detections, i, detector.class_names)
            if name not in equipment_colors:
                continue
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            confidence = float(detections.confidence[i])
            tracker_text = ""
            if detections.tracker_id is not None:
                tracker_text = f":{int(detections.tracker_id[i])}"
            color = equipment_colors[name]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{name}{tracker_text} {confidence:.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        active_worker_ids = set()
        for worker in workers:
            worker_id = worker["id"]
            active_worker_ids.add(worker_id)
            if worker_id not in worker_states:
                worker_states[worker_id] = WorkerSafetyState(
                    worker_id=worker_id, first_seen=now, last_seen=now
                )
            state = worker_states[worker_id]

            matched_hook = hook_matches.get(worker_id)
            matched_harness = harness_matches.get(worker_id)
            hook_seen = matched_hook is not None
            harness_seen = matched_harness is not None
            connected_probability = None

            if matched_hook is not None:
                connected_probability = classify_hook_crop(frame_rgb, matched_hook["bbox"])
                hx1, hy1, hx2, hy2 = map(int, matched_hook["bbox"])
                hook_color = (0, 255, 0) if (
                    connected_probability is not None
                    and connected_probability >= CONNECTED_ENTER_THRESHOLD
                ) else (0, 0, 255)
                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), hook_color, 2)
                wcx, wcy = map(int, box_center(worker["bbox"]))
                hcx, hcy = map(int, box_center(matched_hook["bbox"]))
                cv2.line(annotated, (wcx, wcy), (hcx, hcy), hook_color, 2)

            previous = state.status
            state.update(now, connected_probability, hook_seen, harness_seen)
            draw_worker_status(annotated, worker, state)

            # 상태가 DANGER로 전환될 때만 이벤트를 기록합니다.
            if previous != "DANGER" and state.status == "DANGER":
                if now - state.last_alert_time >= CONFIG["alert_cooldown_seconds"]:
                    event = {
                        "video_time_seconds": round(now, 3),
                        "frame": frame_index,
                        "worker_id": worker_id,
                        "status": state.status,
                        "reason": state.reason,
                        "connected_probability_ema": state.ema_connected,
                    }
                    events.append(event)
                    state.last_alert_time = now
                    print("ALERT:", event)

        # 화면에서 사라진 worker 상태는 TTL 후 제거합니다.
        expired_ids = [
            worker_id for worker_id, state in worker_states.items()
            if worker_id not in active_worker_ids
            and now - state.last_seen >= CONFIG["worker_track_ttl_seconds"]
        ]
        for worker_id in expired_ids:
            del worker_states[worker_id]

        cv2.putText(
            annotated,
            f"time {now:.1f}s | workers {len(workers)} | alerts {len(events)}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(annotated)

        frame_index += 1
        if frame_index % max(1, round(fps * 5)) == 0:
            percent = 100 * frame_index / max(total_frames, 1)
            elapsed = time.time() - start_wall_time
            print(f"{frame_index}/{total_frames} ({percent:.1f}%) elapsed={elapsed/60:.1f}m")
finally:
    cap.release()
    writer.release()

event_columns = [
    "video_time_seconds", "frame", "worker_id", "status", "reason",
    "connected_probability_ema",
]
pd.DataFrame(events, columns=event_columns).to_csv(CONFIG["event_log_csv"], index=False)
print("Completed video:", output_path)
print("Event log:", CONFIG["event_log_csv"])
print("Total danger events:", len(events))


# %% [markdown]
# ## 실행 후 반드시 확인할 항목
#
# 1. RF-DETR가 출력한 `class_names`가 코드의 worker/harness/hook 이름과 일치하는가?
# 2. Test의 False SAFE가 몇 장인가?
# 3. 영상에서 DANGER까지 걸리는 시간이 적절한가?
# 4. 시간당 오경보 수와 깜빡임 횟수가 줄었는가?
# 5. 연속 프레임 데이터가 train/validation/test에 섞이지 않았는가?
