"""COCO segmentation 라벨을 기준으로 안전고리와 주변 맥락을 crop하는 전처리 코드.

원작성자(노션): 조하민
노션 완료 순서: 06 - 이미지 crop
"""

from pathlib import Path
import argparse
import json

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_path(split_dir, file_name):
    direct_path = split_dir / file_name
    if direct_path.exists():
        return direct_path

    matches = list(split_dir.rglob(Path(file_name).name))
    if matches:
        return matches[0]

    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def segmentation_to_points(segmentation):
    if not segmentation:
        return []

    # COCO polygon segmentation: [[x1, y1, x2, y2, ...], ...]
    if isinstance(segmentation, list):
        polygons = []
        for polygon in segmentation:
            if not polygon or len(polygon) < 6:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            polygons.append(points)
        return polygons

    # RLE segmentation is not expected from Roboflow polygon exports.
    return []


def bbox_from_polygons(polygons, fallback_bbox=None):
    points = [point for polygon in polygons for point in polygon]
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    if fallback_bbox:
        x, y, w, h = fallback_bbox
        return x, y, x + w, y + h

    return None


def expand_bbox(bbox, image_w, image_h, scale=3.0, min_pad=100):
    x_min, y_min, x_max, y_max = bbox
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


def bbox_area(bbox):
    left, top, right, bottom = bbox
    return max(0, right - left) * max(0, bottom - top)


def bbox_intersection_area(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return bbox_area((left, top, right, bottom))


def bbox_distance(a, b):
    """Pixel distance between two boxes. Returns 0 when they overlap/touch."""
    horizontal_gap = max(a[0] - b[2], b[0] - a[2], 0)
    vertical_gap = max(a[1] - b[3], b[1] - a[3], 0)
    return (horizontal_gap**2 + vertical_gap**2) ** 0.5


def classify_hook_crop(
    hook_bbox,
    crop_box,
    lifeline_annotations,
    min_lifeline_crop_ratio=0.15,
    near_px=40,
    near_hook_ratio=0.75,
):
    hook_w = hook_bbox[2] - hook_bbox[0]
    hook_h = hook_bbox[3] - hook_bbox[1]
    near_threshold = max(near_px, max(hook_w, hook_h) * near_hook_ratio)

    partial_lifelines = []
    visible_lifelines = []
    near_lifelines = []

    for annotation in lifeline_annotations:
        polygons = segmentation_to_points(annotation.get("segmentation"))
        lifeline_bbox = bbox_from_polygons(polygons, annotation.get("bbox"))
        if not lifeline_bbox:
            continue

        lifeline_area = bbox_area(lifeline_bbox)
        if lifeline_area == 0:
            continue

        intersection = bbox_intersection_area(lifeline_bbox, crop_box)
        crop_ratio = intersection / lifeline_area
        if crop_ratio <= 0:
            continue

        distance = bbox_distance(hook_bbox, lifeline_bbox)
        lifeline_info = {
            "annotation_id": annotation["id"],
            "bbox_xyxy": list(lifeline_bbox),
            "crop_ratio": crop_ratio,
            "distance_to_hook_px": distance,
        }

        partial_lifelines.append(lifeline_info)
        if crop_ratio >= min_lifeline_crop_ratio:
            visible_lifelines.append(lifeline_info)
            if distance <= near_threshold:
                near_lifelines.append(lifeline_info)

    if near_lifelines:
        return {
            "pseudo_label": "connected",
            "reason": "near_lifeline_in_hook_crop",
            "near_threshold_px": near_threshold,
            "lifelines": near_lifelines,
        }

    if visible_lifelines:
        return {
            "pseudo_label": "unknown",
            "reason": "lifeline_visible_but_not_near_this_hook",
            "near_threshold_px": near_threshold,
            "lifelines": visible_lifelines,
        }

    if partial_lifelines:
        return {
            "pseudo_label": "unknown",
            "reason": "lifeline_only_partially_inside_crop",
            "near_threshold_px": near_threshold,
            "lifelines": partial_lifelines,
        }

    return {
        "pseudo_label": "unconnected",
        "reason": "no_lifeline_in_hook_crop",
        "near_threshold_px": near_threshold,
        "lifelines": [],
    }


def transform_polygon_to_crop(points, crop_box, output_size=None):
    left, top, right, bottom = crop_box
    crop_w = right - left
    crop_h = bottom - top

    scale_x = 1.0
    scale_y = 1.0
    if output_size:
        scale_x = output_size[0] / crop_w
        scale_y = output_size[1] / crop_h

    return [((x - left) * scale_x, (y - top) * scale_y) for x, y in points]


def draw_polygons(image, polygons_by_name):
    draw = ImageDraw.Draw(image)
    colors = {
        "hook": (255, 255, 0),
        "lifeline": (0, 255, 255),
        "lanyard": (0, 255, 0),
        "harness": (255, 128, 0),
        "worker": (255, 0, 255),
    }
    for class_name, polygons in polygons_by_name:
        color = colors.get(class_name, (255, 0, 0))
        for polygon in polygons:
            if len(polygon) >= 2:
                draw.line(polygon + [polygon[0]], fill=color, width=2)


def crop_dataset(
    dataset_dir,
    output_dir,
    hook_name,
    lifeline_name,
    context_names,
    scale,
    min_pad,
    resize,
    draw_overlay,
    save_by_label,
    min_lifeline_crop_ratio,
    near_px,
    near_hook_ratio,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    metadata = []

    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir / split
        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            continue

        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {category["id"]: category["name"] for category in coco["categories"]}
        hook_category_ids = {category_id for category_id, name in categories.items() if name == hook_name}
        lifeline_category_ids = {category_id for category_id, name in categories.items() if name == lifeline_name}
        if not hook_category_ids:
            raise ValueError(f"Could not find class '{hook_name}' in {annotation_path}")
        if not lifeline_category_ids:
            raise ValueError(f"Could not find class '{lifeline_name}' in {annotation_path}")

        images = {image["id"]: image for image in coco["images"]}
        annotations_by_image = {}
        for annotation in coco["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image_id, image_info in images.items():
            image_path = find_image_path(split_dir, image_info["file_name"])
            if not image_path:
                print(f"Skip missing image: {split}/{image_info['file_name']}")
                continue

            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size
            annotations = annotations_by_image.get(image_id, [])
            hook_annotations = [
                annotation for annotation in annotations if annotation["category_id"] in hook_category_ids
            ]
            lifeline_annotations = [
                annotation for annotation in annotations if annotation["category_id"] in lifeline_category_ids
            ]

            for hook_index, hook_annotation in enumerate(hook_annotations, start=1):
                hook_polygons = segmentation_to_points(hook_annotation.get("segmentation"))
                hook_bbox = bbox_from_polygons(hook_polygons, hook_annotation.get("bbox"))
                if not hook_bbox:
                    continue

                crop_box = expand_bbox(hook_bbox, image_w, image_h, scale=scale, min_pad=min_pad)
                crop = image.crop(crop_box)
                original_crop_size = crop.size
                label_info = classify_hook_crop(
                    hook_bbox,
                    crop_box,
                    lifeline_annotations,
                    min_lifeline_crop_ratio=min_lifeline_crop_ratio,
                    near_px=near_px,
                    near_hook_ratio=near_hook_ratio,
                )

                transformed = []
                for annotation in annotations:
                    class_name = categories[annotation["category_id"]]
                    if class_name != hook_name and class_name not in context_names:
                        continue

                    polygons = segmentation_to_points(annotation.get("segmentation"))
                    crop_polygons = [
                        transform_polygon_to_crop(polygon, crop_box, output_size=resize)
                        for polygon in polygons
                    ]
                    transformed.append({"class_name": class_name, "polygons": crop_polygons})

                if resize:
                    crop = crop.resize(resize, Image.Resampling.LANCZOS)

                if draw_overlay:
                    draw_polygons(
                        crop,
                        [(item["class_name"], item["polygons"]) for item in transformed],
                    )

                output_name = f"{Path(image_info['file_name']).stem}_hook_{hook_index:02d}.jpg"
                if save_by_label:
                    output_path = split_output_dir / label_info["pseudo_label"] / output_name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = split_output_dir / output_name
                crop.save(output_path, quality=95)

                metadata.append(
                    {
                        "split": split,
                        "source_image": image_info["file_name"],
                        "output_image": str(output_path),
                        "hook_annotation_id": hook_annotation["id"],
                        "hook_bbox_xyxy": list(hook_bbox),
                        "crop_box_xyxy": list(crop_box),
                        "original_crop_size": list(original_crop_size),
                        "resized_to": list(resize) if resize else None,
                        "classes_in_crop": sorted({item["class_name"] for item in transformed}),
                        "pseudo_label": label_info["pseudo_label"],
                        "pseudo_label_reason": label_info["reason"],
                        "near_threshold_px": label_info["near_threshold_px"],
                        "matched_lifelines": label_info["lifelines"],
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "crop_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path, len(metadata)


def parse_resize(value):
    if not value:
        return None
    width, height = value.lower().split("x")
    return int(width), int(height)


def main():
    parser = argparse.ArgumentParser(description="Crop hook-centered images from COCO segmentation datasets.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root containing train/valid/test folders")
    parser.add_argument("--output-dir", default="hook_coco_crops", help="Directory to save crop images")
    parser.add_argument("--hook-name", default="hook", help="Class name to crop around")
    parser.add_argument("--lifeline-name", default="lifeline", help="Class name used as connected lifeline")
    parser.add_argument(
        "--context-names",
        default="lifeline,lanyard,harness",
        help="Comma-separated classes to keep in transformed metadata/overlay",
    )
    parser.add_argument("--scale", type=float, default=3.0, help="Crop size multiplier around hook bbox")
    parser.add_argument("--min-pad", type=int, default=100, help="Minimum pixel padding around hook bbox")
    parser.add_argument("--resize", type=parse_resize, help="Optional fixed output size, e.g. 224x224 or 320x320")
    parser.add_argument("--draw-overlay", action="store_true", help="Draw hook/context polygons on crop")
    parser.add_argument("--flat-output", action="store_true", help="Do not split output folders by pseudo-label")
    parser.add_argument(
        "--min-lifeline-crop-ratio",
        type=float,
        default=0.15,
        help="Minimum fraction of a lifeline bbox that must be inside the hook crop",
    )
    parser.add_argument(
        "--near-px",
        type=float,
        default=40,
        help="Minimum pixel threshold for hook-lifeline distance",
    )
    parser.add_argument(
        "--near-hook-ratio",
        type=float,
        default=0.75,
        help="Also allow hook-lifeline distance up to this ratio of hook bbox size",
    )
    args = parser.parse_args()

    context_names = {name.strip() for name in args.context_names.split(",") if name.strip()}
    metadata_path, count = crop_dataset(
        args.dataset_dir,
        args.output_dir,
        args.hook_name,
        args.lifeline_name,
        context_names,
        args.scale,
        args.min_pad,
        args.resize,
        args.draw_overlay,
        not args.flat_output,
        args.min_lifeline_crop_ratio,
        args.near_px,
        args.near_hook_ratio,
    )

    print(f"Saved {count} crop(s)")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
