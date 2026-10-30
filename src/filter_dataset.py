"""Filter a Roboflow YOLO detection dataset to the requested ripeness classes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError


TARGET_NAMES = (
    "Apple Unripe",
    "Apple Ripe",
    "Apple Overripe",
    "Banana Unripe",
    "Banana Ripe",
    "Banana Overripe",
    "Mango Unripe",
    "Mango Ripe",
    "Mango Overripe",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def load_source_names(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError(f"Không tìm thấy danh sách names trong {config_path}")

    names = config["names"]
    if isinstance(names, dict):
        try:
            class_ids = sorted(int(class_id) for class_id in names)
            if class_ids != list(range(len(class_ids))):
                raise ValueError("ID trong names phải liên tục từ 0")
            result = [str(names.get(class_id, names.get(str(class_id)))) for class_id in class_ids]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Danh sách names không hợp lệ trong {config_path}: {error}") from error
    elif isinstance(names, list):
        result = [str(name) for name in names]
    else:
        raise ValueError(f"names phải là list hoặc mapping trong {config_path}")

    declared_count = config.get("nc")
    if declared_count is not None and declared_count != len(result):
        raise ValueError(f"nc={declared_count} không khớp {len(result)} tên lớp trong {config_path}")
    missing = [name for name in TARGET_NAMES if name not in result]
    if missing:
        raise ValueError(
            "Metadata nguồn thiếu tên lớp mục tiêu chính xác; dừng để không tự suy đoán: "
            + ", ".join(missing)
        )
    return result


def validate_box(fields: list[str], names: list[str]) -> tuple[int, list[str]]:
    if len(fields) != 5:
        is_polygon = len(fields) >= 7 and len(fields) % 2 == 1
        reason = "unsupported_yolo_segmentation_annotation" if is_polygon else "invalid_field_count"
        raise ValueError(reason)
    try:
        class_id = int(fields[0])
    except ValueError as error:
        raise ValueError("invalid_class_id") from error
    if str(class_id) != fields[0] or not 0 <= class_id < len(names):
        raise ValueError("unknown_class_id")

    try:
        coordinates = [float(value) for value in fields[1:]]
    except ValueError as error:
        raise ValueError("invalid_coordinates") from error
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("invalid_coordinates")
    center_x, center_y, width, height = coordinates
    if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError("coordinates_out_of_range")
    if center_x - width / 2 < 0 or center_x + width / 2 > 1:
        raise ValueError("box_exceeds_image_bounds")
    if center_y - height / 2 < 0 or center_y + height / 2 > 1:
        raise ValueError("box_exceeds_image_bounds")
    return class_id, fields[1:]


def image_is_readable(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_output_path(requested: Path, explicit: bool) -> Path:
    if not requested.exists():
        return requested
    if explicit:
        raise FileExistsError(f"Đường dẫn đầu ra đã tồn tại, không ghi đè: {requested}")
    suffix = 2
    while True:
        candidate = requested.with_name(f"{requested.name}-{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def filter_dataset(source: Path, requested_output: Path, explicit_output: bool = False) -> Path:
    source = source.resolve()
    output = unique_output_path(requested_output.resolve(), explicit_output)
    if not source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset nguồn: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Thư mục đầu ra và nguồn không được lồng nhau")

    config_path = source / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình nguồn: {config_path}")
    source_names = load_source_names(config_path)
    target_ids = {name: index for index, name in enumerate(TARGET_NAMES)}
    source_mapping = {
        class_id: target_ids[name] for class_id, name in enumerate(source_names) if name in target_ids
    }

    split_names: dict[str, str] = {}
    for source_split, output_split in SOURCE_SPLITS.items():
        if source_split not in {"val", "valid"} and (source / source_split / "images").is_dir():
            split_names[source_split] = output_split
    if "train" not in split_names:
        raise FileNotFoundError(f"Không tìm thấy tập train/images trong {source}")
    if "val" not in split_names.values():
        validation_split = "valid" if (source / "valid" / "images").is_dir() else "val"
        if (source / validation_split / "images").is_dir():
            split_names[validation_split] = "val"

    anomalies: list[dict[str, Any]] = []
    excluded_images: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    kept_records: list[dict[str, Any]] = []
    box_counts: Counter[str] = Counter()
    source_image_counts: Counter[str] = Counter()
    excluded_reason_counts: Counter[str] = Counter()
    source_label_paths: set[Path] = set()

    for source_split, output_split in split_names.items():
        image_root = source / source_split / "images"
        label_root = source / source_split / "labels"
        images = sorted(
            (path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.relative_to(image_root).as_posix().casefold(),
        )
        source_image_counts[source_split] = len(images)
        split_reports[output_split] = {
            "source_split": source_split,
            "source_images": len(images),
            "kept_images": 0,
            "excluded_images": 0,
            "source_annotation_rows": 0,
            "kept_boxes": 0,
            "removed_boxes": 0,
            "removed_box_reasons": {},
            "classes": {
                name: {"images": 0, "objects": 0} for name in TARGET_NAMES
            },
        }
        for image_path in images:
            relative_image = image_path.relative_to(image_root)
            label_path = (label_root / relative_image).with_suffix(".txt")
            reasons: list[str] = []
            if not image_is_readable(image_path):
                reasons.append("corrupt_image")
                anomalies.append({"split": source_split, "file": str(image_path), "reason": "corrupt_image"})
            if not label_path.is_file():
                reasons.append("missing_label")
                anomalies.append({"split": source_split, "file": str(label_path), "reason": "missing_label"})
            else:
                source_label_paths.add(label_path.resolve())

            kept_lines: list[str] = []
            kept_class_ids: Counter[int] = Counter()
            image_box_reasons: Counter[str] = Counter()
            if not reasons or reasons == ["corrupt_image"]:
                if label_path.is_file():
                    raw_text = label_path.read_text(encoding="utf-8-sig", errors="replace")
                    if not raw_text.strip():
                        reasons.append("empty_label")
                        anomalies.append({"split": source_split, "file": str(label_path), "reason": "empty_label"})
                    else:
                        for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
                            fields = raw_line.split()
                            if not fields:
                                continue
                            split_reports[output_split]["source_annotation_rows"] += 1
                            box_counts["source_rows"] += 1
                            try:
                                source_id, coordinates = validate_box(fields, source_names)
                            except ValueError as error:
                                reason = str(error)
                                image_box_reasons[reason] += 1
                                anomalies.append({
                                    "split": source_split,
                                    "file": str(label_path),
                                    "line": line_number,
                                    "reason": reason,
                                    "content": raw_line,
                                })
                                continue
                            if source_id not in source_mapping:
                                image_box_reasons["non_target_class"] += 1
                                continue
                            target_id = source_mapping[source_id]
                            kept_lines.append(f"{target_id} {' '.join(coordinates)}")
                            kept_class_ids[target_id] += 1

            for reason, count in image_box_reasons.items():
                box_counts[reason] += count
                split_reports[output_split]["removed_boxes"] += count
            split_reports[output_split]["removed_box_reasons"].update(
                Counter(split_reports[output_split]["removed_box_reasons"]) + image_box_reasons
            )

            if not kept_lines and not reasons:
                reasons.append("no_target_annotations")
            if reasons:
                excluded_images.append({
                    "split": source_split,
                    "image": str(image_path),
                    "reasons": reasons,
                })
                split_reports[output_split]["excluded_images"] += 1
                excluded_reason_counts.update(reasons)
                continue

            output_relative_image = Path("images") / output_split / relative_image
            output_relative_label = Path("labels") / output_split / relative_image.with_suffix(".txt")
            kept_records.append({
                "source_split": source_split,
                "output_split": output_split,
                "image_path": image_path,
                "label_path": label_path,
                "relative_image": output_relative_image,
                "relative_label": output_relative_label,
                "label_text": "\n".join(kept_lines) + "\n",
                "class_ids": kept_class_ids,
            })
            split_reports[output_split]["kept_images"] += 1
            split_reports[output_split]["kept_boxes"] += len(kept_lines)
            box_counts["kept"] += len(kept_lines)
            for class_id, count in kept_class_ids.items():
                class_name = TARGET_NAMES[class_id]
                class_stats = split_reports[output_split]["classes"][class_name]
                class_stats["images"] += 1
                class_stats["objects"] += count

        if label_root.is_dir():
            for label_path in sorted(label_root.rglob("*.txt")):
                if label_path.resolve() not in source_label_paths:
                    anomalies.append({
                        "split": source_split,
                        "file": str(label_path),
                        "reason": "orphan_label_without_matching_image",
                    })

    split_priority = {"train": 0, "val": 1, "test": 2}
    duplicate_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in kept_records:
        duplicate_candidates[sha256_file(record["image_path"])].append(record)
    deduplicated_records: list[dict[str, Any]] = []
    duplicate_groups: list[list[dict[str, str]]] = []
    for records in duplicate_candidates.values():
        ordered_records = sorted(records, key=lambda item: split_priority[item["output_split"]])
        kept_splits: set[str] = set()
        group: list[dict[str, str]] = []
        for record in ordered_records:
            split = record["output_split"]
            item = {"split": split, "image": record["relative_image"].as_posix()}
            if kept_splits and split not in kept_splits:
                box_count = sum(record["class_ids"].values())
                excluded_images.append({
                    "split": record["source_split"],
                    "image": str(record["image_path"]),
                    "reasons": ["cross_split_duplicate"],
                })
                excluded_reason_counts["cross_split_duplicate"] += 1
                split_reports[split]["excluded_images"] += 1
                split_reports[split]["kept_images"] -= 1
                split_reports[split]["kept_boxes"] -= box_count
                split_reports[split]["removed_boxes"] += box_count
                split_reports[split]["removed_box_reasons"]["cross_split_duplicate"] = (
                    split_reports[split]["removed_box_reasons"].get("cross_split_duplicate", 0) + box_count
                )
                box_counts["cross_split_duplicate"] += box_count
                for class_id, count in record["class_ids"].items():
                    class_stats = split_reports[split]["classes"][TARGET_NAMES[class_id]]
                    class_stats["images"] -= 1
                    class_stats["objects"] -= count
                group.append({**item, "action": "excluded"})
            else:
                kept_splits.add(split)
                deduplicated_records.append(record)
                group.append({**item, "action": "kept"})
        if len(group) > 1:
            duplicate_groups.append(group)
    kept_records = deduplicated_records

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        output_names: dict[str, list[str]] = defaultdict(list)
        hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
        output_label_ids: set[int] = set()
        output_annotation_errors: list[dict[str, str]] = []
        for record in kept_records:
            output_image = staging / record["relative_image"]
            output_label = staging / record["relative_label"]
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["image_path"], output_image)
            output_label.write_text(record["label_text"], encoding="utf-8")
            relative_image = record["relative_image"].as_posix()
            relative_label = record["relative_label"].as_posix()
            output_names[record["output_split"]].append(relative_image)
            digest = hashlib.sha256(output_image.read_bytes()).hexdigest()
            hashes[digest].append({"split": record["output_split"], "image": relative_image})
            for line_number, line in enumerate(record["label_text"].splitlines(), start=1):
                try:
                    class_id, _ = validate_box(line.split(), list(TARGET_NAMES))
                    output_label_ids.add(class_id)
                except ValueError as error:
                    output_annotation_errors.append({
                        "file": relative_label,
                        "line": str(line_number),
                        "reason": str(error),
                    })

        duplicate_groups = [
            entries for entries in hashes.values()
            if len({entry["split"] for entry in entries}) > 1
        ]
        final_yaml: dict[str, Any] = {
            "path": output.as_posix(),
            "train": "images/train",
            "nc": len(TARGET_NAMES),
            "names": list(TARGET_NAMES),
        }
        if "val" in split_reports:
            final_yaml["val"] = "images/val"
        if "test" in split_reports:
            final_yaml["test"] = "images/test"
        (staging / "data.yaml").write_text(
            yaml.safe_dump(final_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        source_images = sum(source_image_counts.values())
        kept_images = len(kept_records)
        box_counts["removed"] = box_counts["source_rows"] - box_counts["kept"]
        expected_image_paths = {record["relative_image"].as_posix() for record in kept_records}
        expected_label_paths = {record["relative_label"].as_posix() for record in kept_records}
        actual_image_paths = {
            path.relative_to(staging).as_posix()
            for path in (staging / "images").rglob("*") if path.is_file()
        }
        actual_label_paths = {
            path.relative_to(staging).as_posix()
            for path in (staging / "labels").rglob("*.txt")
        }
        quality_checks = {
            "exactly_nine_target_classes_in_yaml": len(final_yaml["names"]) == 9 and tuple(final_yaml["names"]) == TARGET_NAMES,
            "all_output_class_ids_in_range": output_label_ids.issubset(set(range(9))) and not output_annotation_errors,
            "all_output_boxes_valid_yolo": not output_annotation_errors,
            "output_labels_match_output_images": (
                expected_image_paths == actual_image_paths
                and expected_label_paths == actual_label_paths
                and len(expected_image_paths) == len(expected_label_paths)
            ),
            "all_kept_images_readable": all(image_is_readable(staging / path) for paths in output_names.values() for path in paths),
            "no_cross_split_duplicate_images": all(
                len({entry["split"] for entry in group if entry["action"] == "kept"}) <= 1
                for group in duplicate_groups
            ),
            "source_anomalies_reported": True,
            "all_target_classes_have_objects": all(
                sum(split_reports[split]["classes"][name]["objects"] for split in split_reports) > 0
                for name in TARGET_NAMES
            ),
        }
        for split, report in split_reports.items():
            report["removed_box_reasons"] = dict(report["removed_box_reasons"])

        empty_classes = [
            name for name in TARGET_NAMES
            if sum(split_reports[split]["classes"][name]["objects"] for split in split_reports) == 0
        ]
        report_data = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(source),
            "output_dataset": str(output),
            "source_image_count": source_images,
            "kept_image_count": kept_images,
            "excluded_image_count": len(excluded_images),
            "excluded_image_reasons": dict(excluded_reason_counts),
            "annotations": {
                "source_rows": box_counts["source_rows"],
                "kept_boxes": box_counts["kept"],
                "removed_boxes": box_counts["removed"],
                "removed_box_reasons": {
                    reason: box_counts[reason]
                    for reason in sorted(box_counts)
                    if reason not in {"source_rows", "kept", "removed"}
                },
            },
            "source_class_mapping": [
                {
                    "source_id": class_id,
                    "source_name": name,
                    "target_id": source_mapping.get(class_id),
                    "action": "mapped" if class_id in source_mapping else "removed",
                }
                for class_id, name in enumerate(source_names)
            ],
            "splits": split_reports,
            "excluded_images": excluded_images,
            "anomalies": anomalies,
            "duplicate_image_groups_across_splits": duplicate_groups,
            "output_annotation_errors": output_annotation_errors,
            "empty_target_classes": empty_classes,
            "quality_checks": quality_checks,
            "quality_passed": all(quality_checks.values()),
            "ultralytics_installed": __import__("importlib.util").util.find_spec("ultralytics") is not None,
        }
        (staging / "filter_report.json").write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        lines = [
            "BÁO CÁO LỌC DATASET FRUIT RIPENESS",
            f"Nguồn: {source}",
            f"Đầu ra: {output}",
            f"Ảnh nguồn: {source_images}",
            f"Ảnh giữ lại: {kept_images}",
            f"Ảnh bị loại: {len(excluded_images)}",
            f"Bounding box: giữ {box_counts['kept']}, loại {box_counts['removed']}",
            f"Annotation bất thường: {len(anomalies)}",
            f"Trạng thái kiểm tra chất lượng: {'ĐẠT' if report_data['quality_passed'] else 'CẢNH BÁO'}",
            "",
            "Ánh xạ lớp nguồn:",
        ]
        lines.extend(
            f"  {entry['source_id']}: {entry['source_name']} -> "
            + (f"{entry['target_id']}: {TARGET_NAMES[entry['target_id']]}" if entry["target_id"] is not None else "loại bỏ")
            for entry in report_data["source_class_mapping"]
        )
        lines.extend(["", "Thống kê theo tập và lớp:"])
        for split, stats in split_reports.items():
            lines.append(
                f"  {split}: {stats['kept_images']}/{stats['source_images']} ảnh, "
                f"{stats['kept_boxes']} đối tượng"
            )
            for name, counts in stats["classes"].items():
                lines.append(f"    {name}: {counts['images']} ảnh, {counts['objects']} đối tượng")
        lines.extend(["", "Kiểm tra toàn vẹn:"])
        lines.extend(f"  {'ĐẠT' if passed else 'KHÔNG ĐẠT'}: {check}" for check, passed in quality_checks.items())
        if empty_classes:
            lines.extend(["", "CẢNH BÁO: lớp không có đối tượng: " + ", ".join(empty_classes)])
        if excluded_images:
            lines.extend(["", "Ảnh bị loại:"])
            lines.extend(
                f"  [{entry['split']}] {entry['image']}: {', '.join(entry['reasons'])}"
                for entry in excluded_images
            )
        if anomalies:
            lines.extend(["", "Annotation/file bất thường:"])
            lines.extend(
                "  " + json.dumps(entry, ensure_ascii=False)
                for entry in anomalies
            )
        (staging / "filter_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Dataset đầu ra: {output}")
    print(f"Ảnh: {source_images} nguồn, {kept_images} giữ lại, {len(excluded_images)} bị loại")
    print(f"Bounding box: {box_counts['kept']} giữ lại, {box_counts['removed']} loại bỏ")
    print(f"Chất lượng: {'ĐẠT' if report_data['quality_passed'] else 'CẢNH BÁO'}")
    print(f"Báo cáo: {output / 'filter_report.txt'}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw"), help="Dataset YOLO nguồn")
    parser.add_argument("--output", type=Path, help="Thư mục đầu ra riêng; mặc định data/processed/fruit-ripeness-yolo26-filtered")
    args = parser.parse_args(argv)
    requested_output = args.output or Path("data/processed/fruit-ripeness-yolo26-filtered")
    try:
        output = filter_dataset(args.source, requested_output, explicit_output=args.output is not None)
    except (FileNotFoundError, FileExistsError, ValueError, yaml.YAMLError) as error:
        print(f"LỖI: {error}", file=sys.stderr)
        return 2
    report = json.loads((output / "filter_report.json").read_text(encoding="utf-8"))
    return 0 if report["quality_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())