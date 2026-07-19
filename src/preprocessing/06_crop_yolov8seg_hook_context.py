"""YOLOv8 segmentation 라벨을 기준으로 안전고리 주변 crop을 생성하는 전처리 코드.

원작성자(노션): 조하민
노션 완료 순서: 06 - 이미지 crop
"""

from pathlib import Path
import argparse
import re

import numpy as np
from PIL import Image, ImageDraw


def load_class_names(data_yaml):
    """Read YOLO data.yaml names without requiring PyYAML."""
    if not data_yaml:
        return None

    text = Path(data_yaml).read_text(encoding="utf-8")

    # names: [hook, harness, ...]
    inline = re.search(r"(?m)^\s*names\s*:\s*\[(.*?)\]\s*$", text)
    if inline:
        return [item.strip().strip("'\"") for item in inline.group(1).split(",")]

    # names:
    #   0: hook
    #   1: harness
    mapping = {}
    in_names = False
    for line in text.splitlines():
        if re.match(r"^\s*names\s*:\s*$", line):
            in_names = True
            continue
        if in_names:
            match = re.match(r"^\s*(\d+)\s*:\s*['\"]?([^'\"]+)['\"]?\s*$", line)
            if match:
                mapping[int(match.group(1))] = match.group(2).strip()
            elif line and not line.startswith((" ", "\t", "-")):
                break

    if mapping:
        return [mapping[i] for i in sorted(mapping)]

    # names:
    #   - hook
    #   - harness
    list_items = []
    in_names = False
    for line in text.splitlines():
        if re.match(r"^\s*names\s*:\s*$", line):
            in_names = True
            continue
        if in_names:
            match = re.match(r"^\s*-\s*['\"]?([^'\"]+)['\"]?\s*$", line)
            if match:
                list_items.append(match.group(1).strip())
            elif line and not line.startswith((" ", "\t", "-")):
                break

    return list_items or None


def parse_yolov8_seg_label(label_path, image_w, image_h):
    objects = []
    for line_no, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 7:
            continue

        class_id = int(float(parts[0]))
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32)
        if len(coords) % 2 != 0:
            raise ValueError(f"Odd number of polygon coordinates at line {line_no}: {label_path}")

        polygon = coords.reshape(-1, 2)
        polygon[:, 0] *= image_w
        polygon[:, 1] *= image_h
        objects.append({"class_id": class_id, "polygon": polygon})

    return objects


def expanded_bbox_from_polygon(polygon, image_w, image_h, scale=2.0, min_pad=40):
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)

    box_w = x_max - x_min
    box_h = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    crop_w = max(box_w * scale, box_w + min_pad * 2)
    crop_h = max(box_h * scale, box_h + min_pad * 2)

    left = int(max(0, round(cx - crop_w / 2)))
    top = int(max(0, round(cy - crop_h / 2)))
    right = int(min(image_w, round(cx + crop_w / 2)))
    bottom = int(min(image_h, round(cy + crop_h / 2)))

    return left, top, right, bottom


def save_hook_crops(image_path, label_path, output_dir, hook_class_id, scale, min_pad, draw_overlay):
    image_path = Path(image_path)
    label_path = Path(label_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size
    objects = parse_yolov8_seg_label(label_path, image_w, image_h)
    hook_objects = [obj for obj in objects if obj["class_id"] == hook_class_id]

    saved = []
    for index, obj in enumerate(hook_objects, start=1):
        left, top, right, bottom = expanded_bbox_from_polygon(
            obj["polygon"], image_w, image_h, scale=scale, min_pad=min_pad
        )

        crop = image.crop((left, top, right, bottom))

        if draw_overlay:
            shifted_polygon = obj["polygon"].copy()
            shifted_polygon[:, 0] -= left
            shifted_polygon[:, 1] -= top
            points = [tuple(point) for point in shifted_polygon.tolist()]
            ImageDraw.Draw(crop).line(points + [points[0]], fill=(255, 255, 0), width=2)

        out_path = output_dir / f"{image_path.stem}_hook_{index:02d}.jpg"
        crop.save(out_path, quality=95)
        saved.append(out_path)

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Crop wider context around hook instances from YOLOv8 segmentation labels."
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--label", required=True, help="Path to YOLOv8 segmentation txt label")
    parser.add_argument("--output-dir", default="hook_crops", help="Directory to save crops")
    parser.add_argument("--data-yaml", help="Optional Roboflow/YOLO data.yaml to find hook class by name")
    parser.add_argument("--hook-class-id", type=int, help="Hook class id. Used if --data-yaml is omitted")
    parser.add_argument("--hook-name", default="hook", help="Class name to find in data.yaml")
    parser.add_argument("--scale", type=float, default=2.5, help="Crop size multiplier around hook bbox")
    parser.add_argument("--min-pad", type=int, default=80, help="Minimum pixel padding around hook bbox")
    parser.add_argument("--draw-overlay", action="store_true", help="Draw hook polygon on saved crop")
    args = parser.parse_args()

    hook_class_id = args.hook_class_id
    if args.data_yaml:
        names = load_class_names(args.data_yaml)
        if not names or args.hook_name not in names:
            raise ValueError(f"Could not find class name '{args.hook_name}' in {args.data_yaml}")
        hook_class_id = names.index(args.hook_name)

    if hook_class_id is None:
        raise ValueError("Provide either --data-yaml or --hook-class-id")

    saved = save_hook_crops(
        args.image,
        args.label,
        args.output_dir,
        hook_class_id,
        args.scale,
        args.min_pad,
        args.draw_overlay,
    )

    print(f"Saved {len(saved)} hook crop(s)")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
