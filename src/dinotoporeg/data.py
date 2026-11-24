from __future__ import annotations

import csv
import json
import math
import random
import re
import struct
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .utils import atomic_json_dump


@dataclass(frozen=True)
class RegistrationRecord:
    dataset: str
    split: str
    sample_id: str
    source_id: str
    optical: Path
    sar: Path
    flow: Path | None
    affine: Path


def read_flo(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(4) != b"PIEH":
            raise ValueError(f"Invalid Middlebury .flo magic: {path}")
        width, height = struct.unpack("<ii", handle.read(8))
        values = np.fromfile(handle, dtype="<f4")
    expected = width * height * 2
    if values.size != expected:
        raise ValueError(f"Unexpected .flo payload in {path}: {values.size} != {expected}")
    return values.reshape(height, width, 2)


def affine_to_dense_flow(affine: torch.Tensor, height: int, width: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=affine.dtype, device=affine.device),
        torch.arange(width, dtype=affine.dtype, device=affine.device),
        indexing="ij",
    )
    target_x = affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]
    target_y = affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]
    return torch.stack((target_x - xx, target_y - yy), dim=0)


def transform_dense_flow(flow: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Conjugate a dense optical->SAR flow by the pixel transform old->new."""
    squeezed = flow.ndim == 3
    work = flow.unsqueeze(0) if squeezed else flow
    matrices = transform.unsqueeze(0) if transform.ndim == 2 else transform
    if matrices.shape[0] == 1 and work.shape[0] > 1:
        matrices = matrices.expand(work.shape[0], -1, -1)
    batch, _, height, width = work.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=work.device, dtype=work.dtype),
        torch.arange(width, device=work.device, dtype=work.dtype),
        indexing="ij",
    )
    new_points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(1, -1, 3).expand(batch, -1, -1)
    inverse = torch.linalg.inv(matrices)
    old_points = torch.einsum("bij,bnj->bni", inverse, new_points)
    grid_x = 2.0 * old_points[..., 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * old_points[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        work,
        torch.stack((grid_x, grid_y), dim=-1).reshape(batch, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).flatten(2).transpose(1, 2)
    old_target = old_points[..., :2] + sampled
    old_target_h = torch.cat((old_target, torch.ones_like(old_target[..., :1])), dim=-1)
    new_target = torch.einsum("bij,bnj->bni", matrices, old_target_h)[..., :2]
    transformed = (new_target - new_points[..., :2]).transpose(1, 2).reshape(batch, 2, height, width)
    return transformed[0] if squeezed else transformed


def read_manifest(root: str | Path, dataset_name: str, split: str) -> list[RegistrationRecord]:
    root = Path(root)
    manifest = root / split / "manifest.csv"
    if not manifest.exists():
        return discover_records(root, dataset_name, split)
    records: list[RegistrationRecord] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            base = root / split
            records.append(
                RegistrationRecord(
                    dataset=dataset_name,
                    split=split,
                    sample_id=str(row["sample_id"]),
                    source_id=str(row.get("source_id", "")),
                    optical=base / row["optical"],
                    sar=base / row["sar"],
                    flow=base / row["flow"] if row.get("flow", "").strip() else None,
                    affine=base / row["gt"],
                )
            )
    if not records:
        raise ValueError(f"Empty manifest: {manifest}")
    return records


def discover_records(root: Path, dataset_name: str, split: str) -> list[RegistrationRecord]:
    """Discover the standard pair<ID>_1/_2, gt_<ID>, flow_<ID> layout."""
    base = root / split
    image_dir = base / "image"
    gt_dir = base / "gt"
    flow_dir = base / "flow"
    for directory in (image_dir, gt_dir, flow_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Manifest not found and dataset directory is missing: {directory}"
            )

    pattern = re.compile(r"^pair(.+)_1\.png$", re.IGNORECASE)
    ids = [match.group(1) for path in image_dir.iterdir() if path.is_file() and (match := pattern.match(path.name))]
    if not ids:
        raise FileNotFoundError(
            f"Manifest not found and no pair<ID>_1.png images were found: {manifest_path(root, split)}"
        )

    def sort_key(sample_id: str) -> tuple[int, int | str]:
        return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)

    records: list[RegistrationRecord] = []
    missing: list[Path] = []
    for sample_id in sorted(ids, key=sort_key):
        optical = image_dir / f"pair{sample_id}_1.png"
        sar = image_dir / f"pair{sample_id}_2.png"
        affine = gt_dir / f"gt_{sample_id}.txt"
        flow = flow_dir / f"flow_{sample_id}.flo"
        absent = [path for path in (sar, affine, flow) if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        records.append(
            RegistrationRecord(
                dataset=dataset_name,
                split=split,
                sample_id=sample_id,
                source_id=sample_id,
                optical=optical,
                sar=sar,
                flow=flow,
                affine=affine,
            )
        )
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Incomplete dataset pairs under {base}; missing: {preview}")
    return records


def manifest_path(root: Path, split: str) -> Path:
    return root / split / "manifest.csv"


def create_or_load_split(
    records: Sequence[RegistrationRecord],
    split_dir: str | Path,
    validation_ratio: float,
    group_key: str = "source_id",
) -> tuple[list[RegistrationRecord], list[RegistrationRecord], Path]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    dataset_name = records[0].dataset
    split_dir = Path(split_dir)
    if group_key not in ("source_id", "sample_id"):
        raise ValueError("split_group_key must be source_id or sample_id")
    filename = f"{dataset_name}_group-{group_key}_val{validation_ratio:.3f}.json"
    path = split_dir / filename
    duplicate_manifest_ids = [item for item, count in Counter(record.sample_id for record in records).items() if count > 1]
    if duplicate_manifest_ids:
        raise ValueError(f"Duplicate manifest sample IDs for {dataset_name}: {sorted(duplicate_manifest_ids)[:5]}")
    by_id = {record.sample_id: record for record in records}
    def group_id(record: RegistrationRecord) -> str:
        value = getattr(record, group_key, "").strip()
        return value or record.sample_id
    by_group: dict[str, list[RegistrationRecord]] = {}
    for record in records:
        by_group.setdefault(group_id(record), []).append(record)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        train_ids = payload["train_ids"]
        val_ids = payload["val_ids"]
        train_group_ids = payload["train_group_ids"]
        val_group_ids = payload["val_group_ids"]
        for label, ids in (
            ("train_ids", train_ids), ("val_ids", val_ids),
            ("train_group_ids", train_group_ids), ("val_group_ids", val_group_ids),
        ):
            duplicates = [item for item, count in Counter(ids).items() if count > 1]
            if duplicates:
                raise ValueError(f"Duplicate {label} in saved split {path}: {sorted(duplicates)[:5]}")
        sample_overlap = set(train_ids) & set(val_ids)
        if sample_overlap:
            raise ValueError(f"Train/validation sample overlap in saved split {path}: {sorted(sample_overlap)[:5]}")
        missing = (set(train_ids) | set(val_ids)) - set(by_id)
        if missing:
            raise ValueError(f"Saved split {path} refers to missing sample IDs: {sorted(missing)[:5]}")
        unassigned = set(by_id) - (set(train_ids) | set(val_ids))
        if unassigned:
            warnings.warn(
                f"Saved split {path} leaves {len(unassigned)} of {len(records)} manifest pairs unassigned "
                f"(examples: {sorted(unassigned)[:5]}). Preserving the saved train/validation membership "
                "for checkpoint continuation; these pairs are excluded until an explicit split update.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        group_ids = sorted(by_group)
        random.SystemRandom().shuffle(group_ids)
        val_count = max(1, int(round(len(group_ids) * validation_ratio)))
        val_group_ids = sorted(group_ids[:val_count])
        train_group_ids = sorted(group_ids[val_count:])
        train_ids = sorted(record.sample_id for key in train_group_ids for record in by_group[key])
        val_ids = sorted(record.sample_id for key in val_group_ids for record in by_group[key])
        payload = {
            "dataset": dataset_name,
            "validation_ratio": validation_ratio,
            "split_group_key": group_key,
            "total": len(records),
            "group_count": len(group_ids),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "train_group_ids": train_group_ids,
            "val_group_ids": val_group_ids,
        }
        atomic_json_dump(payload, path)
    train_groups = {group_id(by_id[item]) for item in train_ids}
    val_groups = {group_id(by_id[item]) for item in val_ids}
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Grouped split leakage in {path}: {sorted(overlap)[:10]}")
    if train_groups != set(train_group_ids) or val_groups != set(val_group_ids):
        raise RuntimeError(f"Saved group IDs do not match sample membership in {path}")
    return [by_id[item] for item in train_ids], [by_id[item] for item in val_ids], path


def center_crop_registration_pair(optical, sar, flow, affine, crop_size):
    """Crop both coordinate frames equally; preserve displacement units.

    For new coordinates x'=x-c, A'=A and t'=t+(A-I)c.
    Flow is sliced, never resized. Validity is recomputed by the caller.
    Optional labels support single-pair inference without ground truth.
    """
    if crop_size is None:
        return optical, sar, flow, affine
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if len(crop_size) != 2 or any(int(v) != v or v <= 0 for v in crop_size):
        raise ValueError("center_crop_size must contain two positive integers")
    ch, cw = map(int, crop_size)
    height, width = optical.shape[-2:]
    if sar.shape[-2:] != (height, width) or (flow is not None and flow.shape[-2:] != (height, width)):
        raise ValueError("Optical, SAR and flow sizes must agree before center cropping")
    if ch > height or cw > width:
        raise ValueError(f"Cannot center crop {(height, width)} to {(ch, cw)}; resizing is disabled")
    top, left = (height-ch)//2, (width-cw)//2
    optical = optical[..., top:top+ch, left:left+cw].contiguous()
    sar = sar[..., top:top+ch, left:left+cw].contiguous()
    if flow is not None:
        flow = flow[..., top:top+ch, left:left+cw].contiguous()
    if affine is not None:
        affine = affine.clone()
        offset = affine.new_tensor((left, top))
        affine[:, 2] += affine[:, :2] @ offset - offset
    return optical, sar, flow, affine


class RegistrationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[RegistrationRecord],
        dino_mean: Sequence[float],
        dino_std: Sequence[float],
        augment: dict[str, Any] | None = None,
        use_sar_content_mask: bool = True,
        sar_black_threshold: float = 2.0 / 255.0,
        content_mask: dict[str, Any] | None = None,
        center_crop_size: Sequence[int] | None = None,
    ) -> None:
        self.center_crop_size = center_crop_size
        self.records = list(records)
        self.mean = torch.tensor(dino_mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(dino_std, dtype=torch.float32)[:, None, None]
        self.augment = augment or {"enabled": False}
        self.use_sar_content_mask = use_sar_content_mask
        self.sar_black_threshold = float(sar_black_threshold)
        self.content_mask = dict(content_mask or {"enabled": False})

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _homogeneous(affine: torch.Tensor) -> torch.Tensor:
        matrix = torch.eye(3, dtype=affine.dtype, device=affine.device)
        matrix[:2] = affine
        return matrix

    @staticmethod
    def _warp_forward(image: torch.Tensor, forward: torch.Tensor) -> torch.Tensor:
        """Warp an image by a pixel-coordinate forward transform old -> new."""
        _, height, width = image.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=image.dtype, device=image.device),
            torch.arange(width, dtype=image.dtype, device=image.device),
            indexing="ij",
        )
        output = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
        inverse = torch.linalg.inv(forward)
        source = torch.einsum("ij,hwj->hwi", inverse, output)
        source_x = 2.0 * source[..., 0] / max(width - 1, 1) - 1.0
        source_y = 2.0 * source[..., 1] / max(height - 1, 1) - 1.0
        grid = torch.stack((source_x, source_y), dim=-1).unsqueeze(0)
        return F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0]

    @staticmethod
    def _affine_to_flow(affine: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return affine_to_dense_flow(affine, height, width)

    def _relative_affine_augment(
        self,
        sar: torch.Tensor,
        flow: torch.Tensor,
        affine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probability = float(self.augment.get("relative_affine_probability", 0.0))
        if random.random() >= probability:
            return sar, flow, affine
        _, height, width = sar.shape
        angle = math.radians(random.uniform(
            -float(self.augment.get("relative_rotation_degrees", 0.0)),
            float(self.augment.get("relative_rotation_degrees", 0.0)),
        ))
        scale_jitter = float(self.augment.get("relative_scale", 0.0))
        scale = random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        translation = float(self.augment.get("relative_translation_pixels", 0.0))
        tx = random.uniform(-translation, translation)
        ty = random.uniform(-translation, translation)
        cosine = math.cos(angle) * scale
        sine = math.sin(angle) * scale
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        to_origin = affine.new_tensor(((1.0, 0.0, -center_x), (0.0, 1.0, -center_y), (0.0, 0.0, 1.0)))
        linear = affine.new_tensor(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
        from_origin = affine.new_tensor(((1.0, 0.0, center_x + tx), (0.0, 1.0, center_y + ty), (0.0, 0.0, 1.0)))
        relative = from_origin @ linear @ to_origin
        sar = self._warp_forward(sar, relative)
        _, flow_height, flow_width = flow.shape
        yy, xx = torch.meshgrid(
            torch.arange(flow_height, dtype=flow.dtype, device=flow.device),
            torch.arange(flow_width, dtype=flow.dtype, device=flow.device),
            indexing="ij",
        )
        target = torch.stack((xx + flow[0], yy + flow[1], torch.ones_like(xx)), dim=-1)
        transformed_target = torch.einsum("ij,hwj->hwi", relative, target)
        flow = torch.stack((transformed_target[..., 0] - xx, transformed_target[..., 1] - yy), dim=0)
        affine = (relative @ self._homogeneous(affine))[:2]
        return sar, flow, affine

    def _joint_geometric_augment(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        flow: torch.Tensor,
        affine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, height, width = optical.shape
        affine_h = self._homogeneous(affine)

        if random.random() < float(self.augment.get("horizontal_flip_probability", 0.0)):
            transform = affine.new_tensor(((-1.0, 0.0, width - 1), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
            optical = torch.flip(optical, dims=(-1,))
            sar = torch.flip(sar, dims=(-1,))
            flow = transform_dense_flow(flow, transform)
            affine_h = transform @ affine_h @ torch.linalg.inv(transform)

        if random.random() < float(self.augment.get("vertical_flip_probability", 0.0)):
            transform = affine.new_tensor(((1.0, 0.0, 0.0), (0.0, -1.0, height - 1), (0.0, 0.0, 1.0)))
            optical = torch.flip(optical, dims=(-2,))
            sar = torch.flip(sar, dims=(-2,))
            flow = transform_dense_flow(flow, transform)
            affine_h = transform @ affine_h @ torch.linalg.inv(transform)

        if random.random() < float(self.augment.get("rotate90_probability", 0.0)):
            if height != width:
                raise ValueError("90-degree augmentation currently requires square inputs")
            turns = random.choice((1, 2, 3))
            if turns == 1:
                transform = affine.new_tensor(((0.0, 1.0, 0.0), (-1.0, 0.0, width - 1), (0.0, 0.0, 1.0)))
            elif turns == 2:
                transform = affine.new_tensor(((-1.0, 0.0, width - 1), (0.0, -1.0, height - 1), (0.0, 0.0, 1.0)))
            else:
                transform = affine.new_tensor(((0.0, -1.0, height - 1), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
            optical = torch.rot90(optical, turns, dims=(-2, -1))
            sar = torch.rot90(sar, turns, dims=(-2, -1))
            flow = transform_dense_flow(flow, transform)
            affine_h = transform @ affine_h @ torch.linalg.inv(transform)
        return optical.contiguous(), sar.contiguous(), flow.contiguous(), affine_h[:2].contiguous()

    @staticmethod
    def _average_blur(image: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return image
        return F.avg_pool2d(image.unsqueeze(0), kernel, stride=1, padding=kernel // 2)[0]

    @staticmethod
    def _degrade_resolution(image: torch.Tensor, factor: float) -> torch.Tensor:
        height, width = image.shape[-2:]
        low_size = (max(2, int(round(height * factor))), max(2, int(round(width * factor))))
        low = F.interpolate(image.unsqueeze(0), size=low_size, mode="bilinear", align_corners=True)
        return F.interpolate(low, size=(height, width), mode="bilinear", align_corners=True)[0]

    def _random_erasing(self, image: torch.Tensor, probability: float) -> torch.Tensor:
        if random.random() >= probability:
            return image
        _, height, width = image.shape
        min_fraction = float(self.augment.get("erase_min_fraction", 0.03))
        max_fraction = float(self.augment.get("erase_max_fraction", 0.10))
        erase_h = max(1, int(height * random.uniform(min_fraction, max_fraction)))
        erase_w = max(1, int(width * random.uniform(min_fraction, max_fraction)))
        top = random.randint(0, max(height - erase_h, 0))
        left = random.randint(0, max(width - erase_w, 0))
        image = image.clone()
        image[:, top : top + erase_h, left : left + erase_w] = image.mean(dim=(-2, -1), keepdim=True)
        return image

    @staticmethod
    def _content_mask(
        image: torch.Tensor,
        dark_threshold: float,
        variance_kernel: int,
        variance_threshold: float,
    ) -> torch.Tensor:
        """Keep real dark texture while rejecting near-zero, locally flat padding."""
        gray = image.mean(dim=0, keepdim=True)
        kernel = int(variance_kernel)
        if kernel <= 0 or kernel % 2 != 1:
            raise ValueError("content_mask.variance_kernel must be a positive odd integer")
        padding = kernel // 2
        padded = F.pad(gray.unsqueeze(0), (padding, padding, padding, padding), mode="replicate")
        local_mean = F.avg_pool2d(padded, kernel, stride=1)[0]
        local_square_mean = F.avg_pool2d(padded.square(), kernel, stride=1)[0]
        local_variance = (local_square_mean - local_mean.square()).clamp_min(0.0)
        black_border = (gray.abs() <= float(dark_threshold)) & (
            local_variance <= float(variance_threshold)
        )
        return ~black_border

    def _photometric_augment(self, optical: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augment.get("enabled", False):
            return optical, sar
        brightness = float(self.augment.get("optical_brightness", 0.0))
        contrast = float(self.augment.get("optical_contrast", 0.0))
        if brightness > 0:
            optical = optical * random.uniform(1.0 - brightness, 1.0 + brightness)
        if contrast > 0:
            mean = optical.mean(dim=(-2, -1), keepdim=True)
            optical = (optical - mean) * random.uniform(1.0 - contrast, 1.0 + contrast) + mean
        optical_gamma = float(self.augment.get("optical_gamma", 0.0))
        if optical_gamma > 0:
            optical = optical.clamp_min(1.0e-6).pow(random.uniform(1.0 - optical_gamma, 1.0 + optical_gamma))
        saturation = float(self.augment.get("optical_saturation", 0.0))
        if saturation > 0 and optical.shape[0] == 3:
            gray = optical.mean(dim=0, keepdim=True)
            optical = gray + (optical - gray) * random.uniform(1.0 - saturation, 1.0 + saturation)
        optical_noise = float(self.augment.get("optical_noise_std", 0.0))
        if optical_noise > 0:
            optical = optical + torch.randn_like(optical) * optical_noise
        if random.random() < float(self.augment.get("optical_blur_probability", 0.0)):
            optical = self._average_blur(optical, random.choice((3, 5)))
        if random.random() < float(self.augment.get("optical_resolution_probability", 0.0)):
            optical = self._degrade_resolution(optical, random.uniform(0.5, 0.85))

        sar_contrast = float(self.augment.get("sar_contrast", 0.0))
        if sar_contrast > 0:
            mean = sar.mean(dim=(-2, -1), keepdim=True)
            sar = (sar - mean) * random.uniform(1.0 - sar_contrast, 1.0 + sar_contrast) + mean
        sar_gamma = float(self.augment.get("sar_gamma", 0.0))
        if sar_gamma > 0:
            sar = sar.clamp_min(1.0e-6).pow(random.uniform(1.0 - sar_gamma, 1.0 + sar_gamma))
        speckle_std = float(self.augment.get("sar_log_speckle_std", 0.0))
        if speckle_std > 0:
            sar = sar * torch.exp(torch.randn_like(sar) * speckle_std - 0.5 * speckle_std**2)
        sar_noise = float(self.augment.get("sar_noise_std", 0.0))
        if sar_noise > 0:
            sar = sar + torch.randn_like(sar) * sar_noise
        if random.random() < float(self.augment.get("sar_blur_probability", 0.0)):
            sar = self._average_blur(sar, random.choice((3, 5)))
        if random.random() < float(self.augment.get("sar_resolution_probability", 0.0)):
            sar = self._degrade_resolution(sar, random.uniform(0.5, 0.85))

        erase_probability = float(self.augment.get("random_erasing_probability", 0.0))
        optical = self._random_erasing(optical, erase_probability)
        sar = self._random_erasing(sar, erase_probability)
        return optical.clamp(0.0, 1.0), sar.clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.optical) as image:
            optical_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(record.sar) as image:
            sar_np = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        affine_np = np.loadtxt(record.affine, dtype=np.float32).reshape(2, 3)

        optical = torch.from_numpy(optical_np).permute(2, 0, 1).contiguous()
        sar = torch.from_numpy(sar_np).unsqueeze(0).contiguous()
        affine = torch.from_numpy(affine_np).contiguous()
        if record.flow is None:
            # TXT is already optical->SAR in this image's pixel frame.
            # Generate labels online instead of storing redundant .flo files.
            flow = affine_to_dense_flow(affine, *optical.shape[-2:])
        else:
            flow_np = read_flo(record.flow).astype(np.float32)
            flow = torch.from_numpy(flow_np).permute(2, 0, 1).contiguous()
        if self.augment.get("enabled", False):
            spatial_config = self.augment.get("gdros_spatial", {})
            if spatial_config.get("enabled", False):
                from .spatial_augmentation import GDROSSpatialAugmentor
                optical, sar, flow, affine = GDROSSpatialAugmentor(spatial_config)(
                    optical, sar, flow, affine
                )
            else:
                sar, flow, affine = self._relative_affine_augment(sar, flow, affine)
                optical, sar, flow, affine = self._joint_geometric_augment(optical, sar, flow, affine)

        optical, sar, flow, affine = center_crop_registration_pair(
            optical, sar, flow, affine, self.center_crop_size
        )
        height, width = optical.shape[-2:]
        if flow.shape[-2:] != (height, width):
            raise ValueError(f"Flow/image size mismatch for {record.sample_id}: {flow.shape[-2:]} vs {(height, width)}")
        optical_for_mask = optical.clone()
        sar_for_mask = sar.clone()
        optical, sar = self._photometric_augment(optical, sar)

        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=flow.dtype),
            torch.arange(width, dtype=flow.dtype),
            indexing="ij",
        )
        target_x = xx + flow[0]
        target_y = yy + flow[1]
        valid = (
            (target_x >= 0.0)
            & (target_x <= width - 1)
            & (target_y >= 0.0)
            & (target_y <= height - 1)
        )
        if self.use_sar_content_mask:
            sample_x = target_x.round().long().clamp(0, width - 1)
            sample_y = target_y.round().long().clamp(0, height - 1)
            valid &= sar_for_mask[0, sample_y, sample_x] > self.sar_black_threshold
        valid = valid.unsqueeze(0)

        mask_cfg = self.content_mask
        if bool(mask_cfg.get("enabled", False)):
            optical_content_mask = self._content_mask(
                optical_for_mask,
                float(mask_cfg.get("optical_dark_threshold", 0.01)),
                int(mask_cfg.get("variance_kernel", 7)),
                float(mask_cfg.get("optical_variance_threshold", 5.0e-5)),
            )
            sar_content_mask = self._content_mask(
                sar_for_mask,
                float(mask_cfg.get("sar_dark_threshold", 0.008)),
                int(mask_cfg.get("variance_kernel", 7)),
                float(mask_cfg.get("sar_variance_threshold", 2.0e-5)),
            )
        else:
            optical_content_mask = torch.ones((1, height, width), dtype=torch.bool)
            sar_content_mask = torch.ones((1, height, width), dtype=torch.bool)

        sar_rgb = sar.expand(3, -1, -1)
        # Filling with the pre-normalization DINO mean makes invalid pixels
        # approximately zero after normalization without changing the backbone.
        optical_for_dino = torch.where(optical_content_mask, optical, self.mean)
        sar_for_dino = torch.where(sar_content_mask, sar_rgb, self.mean)
        dino_optical = (optical_for_dino - self.mean) / self.std
        dino_sar = (sar_for_dino - self.mean) / self.std
        return {
            "optical": optical,
            "sar": sar,
            "dino_optical": dino_optical,
            "dino_sar": dino_sar,
            "optical_content_mask": optical_content_mask,
            "sar_content_mask": sar_content_mask,
            "flow": flow,
            "valid": valid,
            "affine": affine,
            "dataset": record.dataset,
            "sample_id": record.sample_id,
            "source_id": record.source_id,
            "optical_path": str(record.optical),
            "sar_path": str(record.sar),
        }


def _loader_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    workers = int(config.get("num_workers", 0))
    return {
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True)),
        "persistent_workers": bool(config.get("persistent_workers", True)) and workers > 0,
    }


def build_train_val_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    data_cfg = config["data"]
    dino_cfg = config["model"]["dino"]
    train_sets: list[RegistrationDataset] = []
    val_sets: list[RegistrationDataset] = []
    split_summary: dict[str, Any] = {}
    for entry in data_cfg["datasets"]:
        records = read_manifest(entry["root"], entry["name"], "training")
        train_records, val_records, split_path = create_or_load_split(
            records,
            data_cfg["split_dir"],
            float(data_cfg.get("validation_ratio", 0.1)),
            str(data_cfg.get("split_group_key", "source_id")),
        )
        common = dict(
            dino_mean=dino_cfg["mean"],
            dino_std=dino_cfg["std"],
            use_sar_content_mask=bool(data_cfg.get("use_sar_content_mask", True)),
            sar_black_threshold=float(data_cfg.get("sar_black_threshold", 2.0 / 255.0)),
            content_mask=data_cfg.get("content_mask"),
            center_crop_size=data_cfg.get("center_crop_size"),
        )
        train_sets.append(RegistrationDataset(train_records, augment=data_cfg.get("augment"), **common))
        validation_common = dict(common)
        validation_common["center_crop_size"] = data_cfg.get(
            "evaluation_crop_size", data_cfg.get("center_crop_size")
        )
        val_sets.append(RegistrationDataset(val_records, augment={"enabled": False}, **validation_common))
        train_sources = {record.source_id or record.sample_id for record in train_records}
        val_sources = {record.source_id or record.sample_id for record in val_records}
        overlap = train_sources & val_sources
        split_group_key = str(data_cfg.get("split_group_key", "source_id"))
        sample_overlap = {r.sample_id for r in train_records} & {r.sample_id for r in val_records}
        if sample_overlap:
            raise RuntimeError(f"sample_id leakage for {entry['name']}: {sorted(sample_overlap)[:10]}")
        if overlap and split_group_key == "source_id":
            raise RuntimeError(f"source_id leakage for {entry['name']}: {sorted(overlap)[:10]}")
        assigned_ids = {record.sample_id for record in (*train_records, *val_records)}
        unassigned_ids = sorted({record.sample_id for record in records} - assigned_ids)
        split_summary[entry["name"]] = {
            "train": len(train_records),
            "validation": len(val_records),
            "train_sources": len(train_sources),
            "validation_sources": len(val_sources),
            "source_intersection": len(overlap),
            "split_group_key": split_group_key,
            "sample_intersection": len(sample_overlap),
            "split_file": str(split_path),
            "manifest_total": len(records),
            "assigned_total": len(assigned_ids),
            "unassigned_count": len(unassigned_ids),
            "unassigned_ids_preview": unassigned_ids[:10],
        }

    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)
    sampler = None
    shuffle = True
    if bool(data_cfg.get("balanced_sampling", False)) and len(train_sets) > 1:
        sample_weights: list[float] = []
        for dataset in train_sets:
            sample_weights.extend([1.0 / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    kwargs = _loader_kwargs(data_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        **kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return train_loader, val_loader, split_summary


def build_test_loader(
    config: dict[str, Any],
    selected_datasets: Iterable[str] | None = None,
    batch_size: int | None = None,
) -> DataLoader:
    data_cfg = config["data"]
    dino_cfg = config["model"]["dino"]
    selected = set(selected_datasets or [entry["name"] for entry in data_cfg["datasets"]])
    datasets: list[RegistrationDataset] = []
    for entry in data_cfg["datasets"]:
        if entry["name"] not in selected:
            continue
        records = read_manifest(entry["root"], entry["name"], "testing")
        datasets.append(
            RegistrationDataset(
                records,
                dino_mean=dino_cfg["mean"],
                dino_std=dino_cfg["std"],
                augment={"enabled": False},
                use_sar_content_mask=bool(data_cfg.get("use_sar_content_mask", True)),
                sar_black_threshold=float(data_cfg.get("sar_black_threshold", 2.0 / 255.0)),
                content_mask=data_cfg.get("content_mask"),
                center_crop_size=data_cfg.get("center_crop_size"),
            )
        )
    if not datasets:
        raise ValueError(f"No configured datasets selected from: {sorted(selected)}")
    dataset: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=int(batch_size or config.get("evaluation", {}).get("batch_size", 1)),
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(data_cfg),
    )
