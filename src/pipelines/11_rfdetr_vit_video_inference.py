"""RF-DETR 분할, ViT 분류와 ByteTrack을 결합한 영상 안전고리 체결 판별 코드.

원작성자(노션): 이용현
노션 완료 순서: 11 - 현재 최종 코드
가중치와 입력 영상은 저장소에 포함하지 않으며 로컬 경로를 설정해 사용한다.
"""

import os
import math
import re
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from PIL import Image
from rfdetr import RFDETRSegLarge
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = Path(os.environ.get("CONSTRUCTION_SAFETY_PROJECT_ROOT", Path.cwd()))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_video_path = str(PROJECT_ROOT / "samples" / "videos" / "input_video.mp4")
output_video_path = str(OUTPUT_DIR / "rfdetr_vit_output.mp4")

# 학습에 사용한 ViT 체크포인트 경로로 변경하세요.
vit_checkpoint_path = str(
    PROJECT_ROOT
    / "models"
    / "classification"
    / "vit"
    / "vit_connected_classifier_best_recall.pt"
)

# 학습 데이터의 class_to_idx가 connected=0인 경우 그대로 사용합니다.
# connected=1로 학습했다면 1로 바꾸세요.
CONNECTED_CLASS_INDEX = 0
NUM_CLASSES = 2


def adapt_dissected_vit_state_dict(state_dict, target_model):
    """분리형 ViT 체크포인트를 torchvision ViT 키 구조로 변환합니다."""
    target_keys = set(target_model.state_dict())
    target_prefix = "vit." if "vit.class_token" in target_keys else ""
    converted = {}
    qkv_parts = {}

    direct_suffix_map = {
        "embeddings.cls_token": "class_token",
        "embeddings.position_embeddings": "encoder.pos_embedding",
        "embeddings.patch_embeddings.projection.weight": "conv_proj.weight",
        "embeddings.patch_embeddings.projection.bias": "conv_proj.bias",
        "layernorm.weight": "encoder.ln.weight",
        "layernorm.bias": "encoder.ln.bias",
    }

    for original_key, value in state_dict.items():
        key = original_key.removeprefix("module.").removeprefix("model.")

        # 이미 현재 모델과 동일한 키이면 그대로 둡니다.
        if key in target_keys:
            converted[key] = value
            continue

        source_suffix = key.removeprefix("vit.")
        if source_suffix in direct_suffix_map:
            converted[target_prefix + direct_suffix_map[source_suffix]] = value
            continue

        layer_match = re.fullmatch(
            r"layers\.(\d+)\.(layernorm_before|layernorm_after|attention\.o_proj|"
            r"attention\.[qkv]_proj|mlp\.fc[12])\.(weight|bias)",
            source_suffix,
        )
        if layer_match:
            layer_index, block_name, parameter_name = layer_match.groups()
            layer_prefix = (
                f"{target_prefix}encoder.layers.encoder_layer_{layer_index}."
            )
            block_map = {
                "layernorm_before": "ln_1",
                "layernorm_after": "ln_2",
                "attention.o_proj": "self_attention.out_proj",
                "mlp.fc1": "mlp.0",
                "mlp.fc2": "mlp.3",
            }

            if block_name in {"attention.q_proj", "attention.k_proj", "attention.v_proj"}:
                projection = block_name.split(".")[1][0]
                destination = layer_prefix + f"self_attention.in_proj_{parameter_name}"
                qkv_parts.setdefault(destination, {})[projection] = value
            else:
                destination = layer_prefix + block_map[block_name] + f".{parameter_name}"
                converted[destination] = value
            continue

        # 분류기 명칭 차이도 자동으로 맞춥니다.
        classifier_suffix = key.removeprefix("vit.")
        if classifier_suffix.startswith("classifier."):
            parameter_name = classifier_suffix.split(".", 1)[1]
            for candidate in (
                f"classifier.{parameter_name}",
                f"head.{parameter_name}",
                f"heads.head.{parameter_name}",
            ):
                if candidate in target_keys:
                    converted[candidate] = value
                    break
            continue

        converted[key] = value

    # torchvision MultiheadAttention은 Q/K/V를 하나의 행렬로 보관합니다.
    for destination, parts in qkv_parts.items():
        missing_parts = {"q", "k", "v"} - parts.keys()
        if missing_parts:
            raise RuntimeError(
                f"{destination} 변환에 필요한 Q/K/V가 부족합니다: {sorted(missing_parts)}"
            )
        converted[destination] = torch.cat(
            [parts["q"], parts["k"], parts["v"]], dim=0
        )

    return converted


# ==========================================
# 1. AI 모델 로드
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 디바이스: {device}")

print("📥 1단계: RF-DETR (Segmentation) 모델 로드 중...")
rfdetr_checkpoint_path = str(
    PROJECT_ROOT
    / "models"
    / "detection"
    / "rfdetr"
    / "checkpoint_best_ema_v2.pth"
)
model = RFDETRSegLarge(pretrain_weights=rfdetr_checkpoint_path)

print("📥 2단계: ViT-B/16 (2-class Classification) 모델 로드 중...")
cls_model = models.vit_b_16(weights=None)
cls_model.heads.head = nn.Linear(cls_model.heads.head.in_features, NUM_CLASSES)

checkpoint = torch.load(vit_checkpoint_path, map_location=device)

# 자주 사용하는 체크포인트 저장 형식을 모두 지원합니다.
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# DataParallel/DistributedDataParallel로 저장한 경우의 접두사를 제거합니다.
state_dict = {
    key.removeprefix("module.").removeprefix("model."): value
    for key, value in state_dict.items()
}

if any("embeddings.cls_token" in key for key in state_dict):
    print("🔄 분리형 ViT 체크포인트를 torchvision 형식으로 변환합니다...")
    state_dict = adapt_dissected_vit_state_dict(state_dict, cls_model)

incompatible = cls_model.load_state_dict(state_dict, strict=False)
if incompatible.missing_keys or incompatible.unexpected_keys:
    raise RuntimeError(
        "ViT 체크포인트 구조가 아직 모델과 일치하지 않습니다.\n"
        f"Missing keys: {incompatible.missing_keys}\n"
        f"Unexpected keys: {incompatible.unexpected_keys}"
    )
cls_model = cls_model.to(device)
cls_model.eval()

# 반드시 ViT 학습 때 사용한 검증/추론 transform과 같아야 합니다.
cls_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ==========================================
# 2. 클래스 맵 및 스마트/안정화 설정값
# ==========================================
RF_DETR_CLASSES = {
    1: "harness",
    2: "hook",
    3: "lanyard",
    4: "lifeline",
    5: "worker",
}

CLASS_THRESHOLDS = {
    1: 0.40,  # harness
    2: 0.70,  # hook: 작은 객체이므로 디버깅 단계에서는 낮게 설정
    3: 0.40,  # lanyard
    4: 0.60,  # lifeline
    5: 0.70,  # worker
}

STRICT_THRESHOLD = 0.70
GRACE_SECONDS = 3.0

HARNESS_MEMORY_FRAMES = 150
SAFE_MEMORY_FRAMES = 60
DANGER_GRACE_FRAMES = 90

tracker = sv.ByteTrack()

worker_memory = {}
worker_prob_history = {}


# ==========================================
# 3. 영상 프레임별 처리 루프
# ==========================================
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"입력 영상을 열 수 없습니다: {input_video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"출력 영상을 생성할 수 없습니다: {output_video_path}")

print(f"🎬 영상 처리를 시작합니다! (해상도: {width}x{height}, FPS: {fps})")
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % 30 == 0 or frame_idx == total_frames:
        progress = (frame_idx / total_frames) * 100 if total_frames else 0
        print(f"⏳ 진행 상황: [{frame_idx}/{total_frames}] ({progress:.1f}%)")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # 1. 객체 탐지 및 클래스별 임계값 필터링
    min_thresh = min(CLASS_THRESHOLDS.values())
    detections = model.predict(images=pil_image, threshold=min_thresh)

    if len(detections) > 0:
        detection_mask = np.array(
            [
                confidence >= CLASS_THRESHOLDS.get(class_id, 0.5)
                for confidence, class_id in zip(
                    detections.confidence, detections.class_id
                )
            ]
        )
        detections = detections[detection_mask]

    if hasattr(detections, "metadata") and isinstance(detections.metadata, dict):
        detections.metadata = {}
    detections = tracker.update_with_detections(detections)

    # 2. 현재 프레임의 작업자 위치 수집 및 메모리 초기화
    active_workers = {}
    if detections.tracker_id is not None:
        for i in range(len(detections.xyxy)):
            if detections.class_id[i] != 5:
                continue

            w_xmin, w_ymin, w_xmax, w_ymax = map(int, detections.xyxy[i])
            w_id = detections.tracker_id[i]

            w_xmin, w_ymin = max(0, w_xmin), max(0, w_ymin)
            w_xmax, w_ymax = min(width, w_xmax), min(height, w_ymax)

            cx = w_xmin + (w_xmax - w_xmin) // 2
            cy = w_ymin + (w_ymax - w_ymin) // 2

            active_workers[w_id] = {
                "bbox": (w_xmin, w_ymin, w_xmax, w_ymax),
                "center": (cx, cy),
                "hook_pos": None,
            }

            if w_id not in worker_memory:
                worker_memory[w_id] = {
                    "harness_left": 0,
                    "safe_left": 0,
                    "danger_count": 0,
                }
            if w_id not in worker_prob_history:
                worker_prob_history[w_id] = deque(maxlen=15)

    for memory in worker_memory.values():
        memory["harness_left"] = max(0, memory["harness_left"] - 1)
        memory["safe_left"] = max(0, memory["safe_left"] - 1)

    annotated_frame = frame.copy()

    # 3. 객체 매칭 (하네스, 후크, 랜야드)
    if detections.tracker_id is not None and active_workers:
        for i in range(len(detections.xyxy)):
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            xmin, ymin, xmax, ymax = map(int, detections.xyxy[i])
            cx = xmin + (xmax - xmin) // 2
            cy = ymin + (ymax - ymin) // 2

            closest_w_id = None
            min_dist = float("inf")
            for w_id, w_info in active_workers.items():
                distance = math.hypot(
                    cx - w_info["center"][0], cy - w_info["center"][1]
                )
                if distance < min_dist:
                    min_dist = distance
                    closest_w_id = w_id

            if closest_w_id is None:
                continue

            if class_id == 1:
                worker_memory[closest_w_id]["harness_left"] = HARNESS_MEMORY_FRAMES

            elif class_id == 3:
                cv2.rectangle(
                    annotated_frame, (xmin, ymin), (xmax, ymax), (255, 100, 0), 2
                )
                cv2.putText(
                    annotated_frame,
                    f"Lanyard {confidence:.2f}",
                    (xmin, max(0, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 100, 0),
                    2,
                )

            elif class_id == 2:
                hook_color = (0, 255, 255)
                cv2.rectangle(
                    annotated_frame,
                    (xmin, ymin),
                    (xmax, ymax),
                    hook_color,
                    3,
                )

                hook_w, hook_h = xmax - xmin, ymax - ymin
                margin_px = int(max(hook_w, hook_h) * 0.5)
                c_xmin = max(0, xmin - margin_px)
                c_ymin = max(0, ymin - margin_px)
                c_xmax = min(width, xmax + margin_px)
                c_ymax = min(height, ymax + margin_px)

                cropped_img = frame_rgb[c_ymin:c_ymax, c_xmin:c_xmax]
                if cropped_img.size > 0:
                    pil_crop = Image.fromarray(cropped_img)
                    input_tensor = cls_transform(pil_crop).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        logits = cls_model(input_tensor)
                        probabilities = F.softmax(logits, dim=1)

                    prob_connected = probabilities[0, CONNECTED_CLASS_INDEX].item()
                    worker_prob_history[closest_w_id].append(prob_connected)
                    avg_prob = float(
                        np.mean(worker_prob_history[closest_w_id])
                    )

                    hook_text = (
                        f"Hook DET:{confidence:.2f} "
                        f"ViT:{prob_connected:.2f} AVG:{avg_prob:.2f}"
                    )
                    cv2.putText(
                        annotated_frame,
                        hook_text,
                        (xmin, max(25, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        hook_color,
                        2,
                        cv2.LINE_AA,
                    )

                    if avg_prob >= STRICT_THRESHOLD:
                        worker_memory[closest_w_id]["safe_left"] = SAFE_MEMORY_FRAMES
                    else:
                        worker_memory[closest_w_id]["safe_left"] = 0

                else:
                    cv2.putText(
                        annotated_frame,
                        f"Hook DET:{confidence:.2f} (invalid crop)",
                        (xmin, max(25, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        hook_color,
                        2,
                        cv2.LINE_AA,
                    )

                active_workers[closest_w_id]["hook_pos"] = (cx, cy)

    # 4. 상태 및 알림 텍스트 렌더링
    for w_id, w_info in active_workers.items():
        memory = worker_memory[w_id]
        w_xmin, w_ymin, w_xmax, w_ymax = w_info["bbox"]

        hx = w_xmin
        hy = max(30, w_ymin - 15)

        if memory["harness_left"] == 0:
            text = f"W:{w_id} - DANGER (No Harness)"
            color = (0, 0, 255)
            memory["danger_count"] = 0

        elif memory["safe_left"] > 0:
            text = f"W:{w_id} - 100% SAFE"
            color = (0, 255, 0)
            memory["danger_count"] = 0
            if w_info["hook_pos"]:
                cv2.line(
                    annotated_frame,
                    w_info["center"],
                    w_info["hook_pos"],
                    color,
                    2,
                    cv2.LINE_AA,
                )

        else:
            memory["danger_count"] += 1
            if memory["danger_count"] >= DANGER_GRACE_FRAMES:
                text = f"W:{w_id} - DANGER (Unconnected)"
                color = (0, 0, 255)
            else:
                time_left = GRACE_SECONDS - (memory["danger_count"] / fps)
                text = f"W:{w_id} - WAIT {max(0, time_left):.1f}s"
                color = (0, 255, 255)

            if w_info["hook_pos"]:
                cv2.line(
                    annotated_frame,
                    w_info["center"],
                    w_info["hook_pos"],
                    color,
                    2,
                    cv2.LINE_AA,
                )

        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            annotated_frame,
            (hx, hy - text_height - 5),
            (hx + text_width, hy + 5),
            color,
            -1,
        )
        cv2.putText(
            annotated_frame,
            text,
            (hx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            annotated_frame,
            (w_xmin, w_ymin),
            (w_xmax, w_ymax),
            color,
            3,
        )

    out.write(annotated_frame)

cap.release()
out.release()
print(f"✅ 영상 저장 완료: {output_video_path}")
