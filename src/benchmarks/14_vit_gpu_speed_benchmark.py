"""RF-DETR 검출과 ViT 분류 파이프라인의 GPU 단계별 처리 속도를 측정합니다.

RF-DETR로 안전고리(hook)를 검출하고 주변 영역을 crop한 뒤,
ViT로 connected/unconnected 상태를 분류하면서 전처리·추론·시각화·저장
단계별 지연 시간과 전체 FPS를 집계합니다.

원작성자: 조은나라
Notion 완료 항목 순서: 14
실행 환경: Google Colab 또는 CUDA 지원 Python 환경
주의: 영상과 모델 가중치는 저장소에 포함하지 않으며 경로를 실행 환경에 맞게 수정해야 합니다.
"""

# ============================================================
# RF-DETR 객체인식
# → hook bbox 기준 80% crop
# → ViT connected / unconnected 분류
# → 단계별 처리속도 측정
# ============================================================

# Colab에서 실행할 경우: !pip install -q transformers

import os
import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from transformers import ViTForImageClassification
from rfdetr import RFDETRSegLarge


# ============================================================
# 1. 경로 설정
# ============================================================

video_path = "/content/drive/MyDrive/1_data/video/cut_video.mp4"

output_video_path = (
    "/content/drive/MyDrive/1_data/video/"
    "hook_pipeline_result_vit_speed.mp4"
)

# RF-DETR 가중치
rfdetr_ckpt_path = (
    "/content/drive/MyDrive/2_weight/"
    "rfdetr_seg_large_output/checkpoint_best_ema.pth"
)

# ViT 분류 모델 가중치
vit_ckpt_path = (
    "/content/drive/MyDrive/2_weight/"
    "all_best_vit_connected_classifier_recall.pt"
)

# crop 저장 폴더
crop_save_dir = Path("/content/video_hook_crops_vit")
crop_save_dir.mkdir(parents=True, exist_ok=True)

# RF-DETR 입력용 임시 프레임
temp_frame_path = "/content/temp_video_frame.jpg"


# ============================================================
# 2. 주요 설정
# ============================================================

# RF-DETR 클래스
# 1: harness
# 2: hook
# 3: lanyard
# 4: lifeline
# 5: worker
HOOK_CLASS_ID = 2

# RF-DETR confidence threshold
DETECTION_THRESHOLD = 0.5

# hook bbox 기준 상하좌우 각각 80% 여백
MARGIN_RATIO = 0.8

# 1이면 모든 프레임에서 객체인식과 분류 수행
FRAME_STRIDE = 1

# 처리속도 측정 시 False 권장
# True이면 crop 이미지 저장 시간까지 전체 속도에 포함됨
SAVE_CROPS = False

# 초기 GPU 워밍업 영향 제외
WARMUP_FRAMES = 10


# ============================================================
# 3. Device 설정
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("사용 device:", device)


def sync_cuda():
    """
    GPU 연산은 비동기로 실행되기 때문에
    정확한 시간 측정을 위해 GPU 연산 종료를 기다린다.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# 4. ViT 모델 생성
# ============================================================

vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    ignore_mismatched_sizes=True
)


# ============================================================
# 5. ViT 가중치 로드
# ============================================================

if not os.path.exists(vit_ckpt_path):
    raise FileNotFoundError(
        f"ViT 가중치를 찾을 수 없습니다:\n{vit_ckpt_path}"
    )

checkpoint = torch.load(
    vit_ckpt_path,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    vit_state_dict = checkpoint["model_state_dict"]

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    vit_state_dict = checkpoint["state_dict"]

elif isinstance(checkpoint, dict) and "model" in checkpoint:
    vit_state_dict = checkpoint["model"]

else:
    vit_state_dict = checkpoint


# DataParallel로 저장된 경우 module. 제거
vit_state_dict = {
    key.replace("module.", ""): value
    for key, value in vit_state_dict.items()
}

vit_model.load_state_dict(
    vit_state_dict,
    strict=True
)

vit_model = vit_model.to(device)
vit_model.eval()

print("ViT 분류 모델 로드 완료")


# ============================================================
# 6. ViT 입력 전처리
# ============================================================

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 7. RF-DETR 모델 로드
# ============================================================

if not os.path.exists(rfdetr_ckpt_path):
    raise FileNotFoundError(
        f"RF-DETR 가중치를 찾을 수 없습니다:\n{rfdetr_ckpt_path}"
    )

rfdetr_model = RFDETRSegLarge(
    pretrain_weights=rfdetr_ckpt_path
)

try:
    rfdetr_model.optimize_for_inference(
        dtype=torch.float16
    )
    print("RF-DETR inference optimization 완료")

except Exception as e:
    print("RF-DETR optimize 생략:", e)

print("RF-DETR 모델 로드 완료")


# ============================================================
# 8. 함수 정의
# ============================================================

def compute_crop_box_from_xyxy(
    xyxy,
    img_w,
    img_h,
    margin_ratio=0.8
):
    """
    hook bbox의 가로·세로 길이를 기준으로
    상하좌우 각각 margin_ratio만큼 확장
    """

    x1, y1, x2, y2 = map(float, xyxy)

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    margin_x = bbox_w * margin_ratio
    margin_y = bbox_h * margin_ratio

    crop_x1 = max(
        0,
        int(np.floor(x1 - margin_x))
    )

    crop_y1 = max(
        0,
        int(np.floor(y1 - margin_y))
    )

    crop_x2 = min(
        img_w,
        int(np.ceil(x2 + margin_x))
    )

    crop_y2 = min(
        img_h,
        int(np.ceil(y2 + margin_y))
    )

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    return (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2
    )


def classify_crop_with_vit(crop_pil):
    """
    클래스 인덱스
    0: connected
    1: unconnected

    반환값에 ViT 전처리 시간과 추론 시간을 함께 포함
    """

    # ----------------------------
    # ViT 입력 전처리 시간
    # ----------------------------
    sync_cuda()
    transform_start = time.perf_counter()

    input_tensor = inference_transform(
        crop_pil
    ).unsqueeze(0).to(device)

    sync_cuda()
    transform_time = (
        time.perf_counter()
        - transform_start
    )

    # ----------------------------
    # ViT 순수 추론 시간
    # ----------------------------
    sync_cuda()
    inference_start = time.perf_counter()

    with torch.inference_mode():

        outputs = vit_model(
            pixel_values=input_tensor
        )

        probabilities = F.softmax(
            outputs.logits,
            dim=1
        )[0]

    sync_cuda()
    inference_time = (
        time.perf_counter()
        - inference_start
    )

    prob_connected = probabilities[0].item()
    prob_unconnected = probabilities[1].item()

    pred_idx = torch.argmax(
        probabilities
    ).item()

    if pred_idx == 0:
        pred_class = "connected"
        pred_conf = prob_connected

    else:
        pred_class = "unconnected"
        pred_conf = prob_unconnected

    return {
        "pred_idx": pred_idx,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "prob_connected": prob_connected,
        "prob_unconnected": prob_unconnected,
        "transform_time": transform_time,
        "inference_time": inference_time
    }


def draw_label(
    img,
    text,
    x,
    y,
    color
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x = int(x)
    y = int(y)

    text_y = max(
        y - 8,
        text_h + 8
    )

    cv2.rectangle(
        img,
        (x, text_y - text_h - 8),
        (x + text_w + 8, text_y + 4),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x + 4, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 9. GPU 워밍업
# ============================================================

print("GPU 워밍업 중...")

dummy_tensor = torch.zeros(
    1,
    3,
    224,
    224,
    device=device
)

with torch.inference_mode():
    for _ in range(5):
        _ = vit_model(
            pixel_values=dummy_tensor
        )

sync_cuda()

print("GPU 워밍업 완료")


# ============================================================
# 10. 동영상 열기
# ============================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(
        f"동영상을 열 수 없습니다:\n{video_path}"
    )

video_fps = cap.get(cv2.CAP_PROP_FPS)

frame_w = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_h = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if video_fps <= 0:
    video_fps = 30.0

Path(output_video_path).parent.mkdir(
    parents=True,
    exist_ok=True
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video_path,
    fourcc,
    video_fps,
    (frame_w, frame_h)
)

if not out.isOpened():
    cap.release()

    raise RuntimeError(
        f"결과 영상을 생성할 수 없습니다:\n"
        f"{output_video_path}"
    )

print("\n영상 정보")
print("video:", video_path)
print("FPS:", video_fps)
print("size:", frame_w, frame_h)
print("total frames:", total_frames)
print("frame stride:", FRAME_STRIDE)
print("RF-DETR threshold:", DETECTION_THRESHOLD)
print("crop 저장:", SAVE_CROPS)


# ============================================================
# 11. 시간 측정 변수
# ============================================================

frame_idx = 0
processed_count = 0
hook_crop_count = 0

connected_count = 0
unconnected_count = 0

last_detections_info = []


# 객체인식 관련
detection_input_times = []
detection_inference_times = []

# crop 관련
crop_coordinate_times = []
crop_image_times = []

# 분류 관련
vit_transform_times = []
vit_inference_times = []
vit_total_times = []

# 저장 및 시각화
crop_save_times = []
visualization_times = []
video_write_times = []

# 전체 파이프라인
full_pipeline_times = []

# 프레임별 hook 개수
hooks_per_frame = []

whole_video_start = time.perf_counter()


# ============================================================
# 12. 동영상 처리
# ============================================================

pbar = tqdm(
    total=total_frames,
    desc="Processing video"
)

while True:

    ret, frame_bgr = cap.read()

    if not ret:
        break

    frame_pipeline_start = time.perf_counter()

    vis_frame = frame_bgr.copy()

    frame_hook_count = 0

    # 지정한 프레임마다 새로 검출·분류
    if frame_idx % FRAME_STRIDE == 0:

        last_detections_info = []

        # ====================================================
        # A. RF-DETR 입력 이미지 준비
        # ====================================================

        input_start = time.perf_counter()

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        Image.fromarray(
            frame_rgb
        ).save(
            temp_frame_path
        )

        detection_input_time = (
            time.perf_counter()
            - input_start
        )

        # ====================================================
        # B. RF-DETR 객체인식
        # ====================================================

        sync_cuda()
        detection_start = time.perf_counter()

        detections = rfdetr_model.predict(
            images=temp_frame_path,
            threshold=DETECTION_THRESHOLD
        )

        sync_cuda()
        detection_inference_time = (
            time.perf_counter()
            - detection_start
        )

        if len(detections) > 0:

            detection_iterator = zip(
                detections.xyxy,
                detections.class_id,
                detections.confidence
            )

            for det_idx, (
                xyxy,
                class_id,
                det_conf
            ) in enumerate(
                detection_iterator,
                start=1
            ):

                class_id = int(class_id)

                # hook 클래스만 사용
                if class_id != HOOK_CLASS_ID:
                    continue

                xyxy = np.asarray(
                    xyxy,
                    dtype=np.float32
                )

                # ====================================================
                # C. crop 좌표 계산
                # ====================================================

                crop_coordinate_start = (
                    time.perf_counter()
                )

                crop_box = compute_crop_box_from_xyxy(
                    xyxy=xyxy,
                    img_w=frame_w,
                    img_h=frame_h,
                    margin_ratio=MARGIN_RATIO
                )

                crop_coordinate_time = (
                    time.perf_counter()
                    - crop_coordinate_start
                )

                if crop_box is None:
                    continue

                (
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2
                ) = crop_box

                # ====================================================
                # D. 실제 crop 이미지 생성
                # ====================================================

                crop_image_start = time.perf_counter()

                crop_bgr = frame_bgr[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2
                ]

                if crop_bgr.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(
                    crop_bgr,
                    cv2.COLOR_BGR2RGB
                )

                crop_pil = Image.fromarray(
                    crop_rgb
                ).convert("RGB")

                crop_image_time = (
                    time.perf_counter()
                    - crop_image_start
                )

                # ====================================================
                # E. ViT 분류
                # ====================================================

                classification_result = (
                    classify_crop_with_vit(
                        crop_pil
                    )
                )

                pred_class = (
                    classification_result[
                        "pred_class"
                    ]
                )

                pred_conf = (
                    classification_result[
                        "pred_conf"
                    ]
                )

                prob_connected = (
                    classification_result[
                        "prob_connected"
                    ]
                )

                prob_unconnected = (
                    classification_result[
                        "prob_unconnected"
                    ]
                )

                vit_transform_time = (
                    classification_result[
                        "transform_time"
                    ]
                )

                vit_inference_time = (
                    classification_result[
                        "inference_time"
                    ]
                )

                vit_total_time = (
                    vit_transform_time
                    + vit_inference_time
                )

                frame_hook_count += 1
                hook_crop_count += 1

                if pred_class == "connected":
                    connected_count += 1
                else:
                    unconnected_count += 1

                # ====================================================
                # F. crop 이미지 저장
                # ====================================================

                crop_save_time = 0.0

                if SAVE_CROPS:

                    crop_save_start = (
                        time.perf_counter()
                    )

                    crop_name = (
                        f"frame{frame_idx:06d}_"
                        f"hook{det_idx}_"
                        f"{pred_class}_"
                        f"c{prob_connected:.3f}_"
                        f"u{prob_unconnected:.3f}.jpg"
                    )

                    crop_pil.save(
                        crop_save_dir / crop_name
                    )

                    crop_save_time = (
                        time.perf_counter()
                        - crop_save_start
                    )

                # ====================================================
                # 워밍업 프레임 이후 시간 저장
                # ====================================================

                if frame_idx >= WARMUP_FRAMES:

                    crop_coordinate_times.append(
                        crop_coordinate_time
                    )

                    crop_image_times.append(
                        crop_image_time
                    )

                    vit_transform_times.append(
                        vit_transform_time
                    )

                    vit_inference_times.append(
                        vit_inference_time
                    )

                    vit_total_times.append(
                        vit_total_time
                    )

                    if SAVE_CROPS:
                        crop_save_times.append(
                            crop_save_time
                        )

                last_detections_info.append({
                    "xyxy": xyxy.copy(),
                    "crop_box": crop_box,
                    "det_conf": float(det_conf),
                    "pred_class": pred_class,
                    "pred_conf": float(pred_conf),
                    "prob_connected": float(
                        prob_connected
                    ),
                    "prob_unconnected": float(
                        prob_unconnected
                    )
                })

        processed_count += 1

        if frame_idx >= WARMUP_FRAMES:

            detection_input_times.append(
                detection_input_time
            )

            detection_inference_times.append(
                detection_inference_time
            )

            hooks_per_frame.append(
                frame_hook_count
            )

    # ========================================================
    # G. 시각화
    # ========================================================

    visualization_start = time.perf_counter()

    for info in last_detections_info:

        x1, y1, x2, y2 = info["xyxy"]

        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2
        ) = info["crop_box"]

        pred_class = info["pred_class"]
        pred_conf = info["pred_conf"]

        prob_connected = info["prob_connected"]
        prob_unconnected = info["prob_unconnected"]

        det_conf = info["det_conf"]

        if pred_class == "connected":
            color = (0, 200, 0)
        else:
            color = (0, 0, 255)

        # 원래 hook bbox
        cv2.rectangle(
            vis_frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 255, 0),
            2
        )

        # 분류에 사용한 crop 영역
        cv2.rectangle(
            vis_frame,
            (crop_x1, crop_y1),
            (crop_x2, crop_y2),
            color,
            2
        )

        label = (
            f"{pred_class} {pred_conf:.2f} "
            f"| C:{prob_connected:.2f} "
            f"U:{prob_unconnected:.2f} "
            f"| hook:{det_conf:.2f}"
        )

        draw_label(
            vis_frame,
            label,
            crop_x1,
            crop_y1,
            color
        )

    visualization_time = (
        time.perf_counter()
        - visualization_start
    )

    # ========================================================
    # H. 결과 영상 저장
    # ========================================================

    video_write_start = time.perf_counter()

    out.write(vis_frame)

    video_write_time = (
        time.perf_counter()
        - video_write_start
    )

    sync_cuda()

    frame_pipeline_time = (
        time.perf_counter()
        - frame_pipeline_start
    )

    if frame_idx >= WARMUP_FRAMES:

        visualization_times.append(
            visualization_time
        )

        video_write_times.append(
            video_write_time
        )

        full_pipeline_times.append(
            frame_pipeline_time
        )

    frame_idx += 1
    pbar.update(1)


# ============================================================
# 13. 종료
# ============================================================

pbar.close()

cap.release()
out.release()

whole_video_time = (
    time.perf_counter()
    - whole_video_start
)

if os.path.exists(temp_frame_path):
    os.remove(temp_frame_path)


# ============================================================
# 14. 결과 계산 함수
# ============================================================

def print_speed_result(
    name,
    values,
    unit="frame"
):
    if len(values) == 0:
        print(f"{name}: 측정값 없음")
        return

    avg_seconds = float(np.mean(values))
    median_seconds = float(np.median(values))
    p95_seconds = float(
        np.percentile(values, 95)
    )

    speed = (
        1.0 / avg_seconds
        if avg_seconds > 0
        else 0.0
    )

    print(f"\n{name}")
    print(
        f"  평균: {avg_seconds * 1000:.3f} ms/{unit}"
    )
    print(
        f"  중앙값: {median_seconds * 1000:.3f} ms/{unit}"
    )
    print(
        f"  P95: {p95_seconds * 1000:.3f} ms/{unit}"
    )
    print(
        f"  처리속도: {speed:.2f} {unit}/s"
    )


# ============================================================
# 15. 단계별 처리속도 출력
# ============================================================

print("\n")
print("=" * 70)
print("객체인식 → crop → 분류 처리속도")
print("=" * 70)

print_speed_result(
    name="1. RF-DETR 입력 이미지 변환 및 저장",
    values=detection_input_times,
    unit="frame"
)

print_speed_result(
    name="2. RF-DETR 객체인식",
    values=detection_inference_times,
    unit="frame"
)

print_speed_result(
    name="3. hook crop 좌표 계산",
    values=crop_coordinate_times,
    unit="crop"
)

print_speed_result(
    name="4. hook crop 이미지 생성",
    values=crop_image_times,
    unit="crop"
)

print_speed_result(
    name="5. ViT 입력 전처리",
    values=vit_transform_times,
    unit="crop"
)

print_speed_result(
    name="6. ViT 순수 분류 추론",
    values=vit_inference_times,
    unit="crop"
)

print_speed_result(
    name="7. ViT 전처리 + 분류",
    values=vit_total_times,
    unit="crop"
)

if SAVE_CROPS:

    print_speed_result(
        name="8. crop 이미지 저장",
        values=crop_save_times,
        unit="crop"
    )

print_speed_result(
    name="9. 결과 박스 및 라벨 시각화",
    values=visualization_times,
    unit="frame"
)

print_speed_result(
    name="10. 결과 영상 저장",
    values=video_write_times,
    unit="frame"
)

print_speed_result(
    name="11. 전체 객체인식-crop-분류 파이프라인",
    values=full_pipeline_times,
    unit="frame"
)


# ============================================================
# 16. 핵심 결과 요약
# ============================================================

print("\n")
print("=" * 70)
print("핵심 결과 요약")
print("=" * 70)

if len(detection_inference_times) > 0:

    avg_detection = np.mean(
        detection_inference_times
    )

    detection_fps = 1 / avg_detection

    print(
        f"객체인식 평균 시간: "
        f"{avg_detection * 1000:.3f} ms/frame"
    )

    print(
        f"객체인식 처리속도: "
        f"{detection_fps:.2f} FPS"
    )


if len(crop_coordinate_times) > 0:

    avg_crop_coordinate = np.mean(
        crop_coordinate_times
    )

    avg_crop_image = np.mean(
        crop_image_times
    )

    avg_crop_total = (
        avg_crop_coordinate
        + avg_crop_image
    )

    print(
        f"\ncrop 평균 시간: "
        f"{avg_crop_total * 1000:.3f} ms/crop"
    )

    print(
        f"crop 처리속도: "
        f"{1 / avg_crop_total:.2f} crops/s"
    )


if len(vit_total_times) > 0:

    avg_vit = np.mean(
        vit_total_times
    )

    print(
        f"\n분류 평균 시간: "
        f"{avg_vit * 1000:.3f} ms/crop"
    )

    print(
        f"분류 처리속도: "
        f"{1 / avg_vit:.2f} crops/s"
    )


if len(full_pipeline_times) > 0:

    avg_pipeline = np.mean(
        full_pipeline_times
    )

    pipeline_fps = 1 / avg_pipeline

    print(
        f"\n전체 파이프라인 평균 시간: "
        f"{avg_pipeline * 1000:.3f} ms/frame"
    )

    print(
        f"전체 파이프라인 처리속도: "
        f"{pipeline_fps:.2f} FPS"
    )

    print(
        f"원본 영상 FPS: "
        f"{video_fps:.2f} FPS"
    )

    realtime_ratio = (
        pipeline_fps / video_fps
    )

    print(
        f"실시간 대비 처리 비율: "
        f"{realtime_ratio:.3f}배"
    )

    if pipeline_fps >= video_fps:
        print("실시간 처리 여부: 가능")

    else:
        print("실시간 처리 여부: 어려움")


if len(hooks_per_frame) > 0:

    print(
        f"\n프레임당 평균 hook 수: "
        f"{np.mean(hooks_per_frame):.3f}개"
    )

    print(
        f"프레임당 최대 hook 수: "
        f"{np.max(hooks_per_frame)}개"
    )


video_duration = (
    frame_idx / video_fps
    if video_fps > 0
    else 0
)

print("\n")
print("=" * 70)
print("처리 완료")
print("=" * 70)

print("저장된 결과 영상:", output_video_path)
print("전체 영상 프레임 수:", frame_idx)
print("실제 객체인식한 프레임 수:", processed_count)
print("검출 및 분류된 hook crop 수:", hook_crop_count)
print("connected 판정 수:", connected_count)
print("unconnected 판정 수:", unconnected_count)

print(
    f"원본 영상 길이: "
    f"{video_duration:.2f}초"
)

print(
    f"실제 전체 실행시간: "
    f"{whole_video_time:.2f}초"
)

if video_duration > 0:

    print(
        f"실행시간 / 영상 길이: "
        f"{whole_video_time / video_duration:.3f}배"
    )
