# -*- coding: utf-8 -*-
"""
ReasonSeg dataset for ReaSeg.

Expected JSON format, e.g.:

[
  {
    "query": "...",
    "outputs": "... [SEG]",
    "image": "73.jpg",
    "json": "73.json"
  }
]

Returned fields:

Qwen2.5-VL branch:
    input_ids
    attention_mask
    labels
    pixel_values
    image_grid_thw

MedSAM branch:
    medsam_image      Tensor[3, image_size, image_size]
    gt_masks          Tensor[N, H, W]
    original_size     (H, W)
    resize_shape      (resized_h, resized_w)
    image_path        str

Backward-compatible aliases:
    images                -> medsam_image
    masks_list            -> gt_masks
    original_size_list    -> original_size
    resize_list           -> [resize_shape]
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

try:
    from model.segment_anything.utils.transforms import ResizeLongestSide
except Exception:
    from ..model.segment_anything.utils.transforms import ResizeLongestSide


class ReaSegReasonSegDataset(Dataset):
    """
    Clean ReasonSeg-style dataset for ReaSeg SFT.

    Args:
        data_path:
            Directory containing train.json / val.json / test.json, or a JSON file.
        tokenizer:
            Qwen tokenizer.
        processor:
            Qwen2.5-VL processor.
        image_size:
            MedSAM/SAM input size, usually 1024.
        max_seq_length:
            Reserved for future truncation. Not forcibly applied here because
            Qwen2.5-VL image tokens and image_grid_thw must stay aligned.
        split:
            train / val / test.
        seg_token:
            Segmentation token text, default "[SEG]".
        seg_token_idx:
            Optional token id for [SEG].
        force_seg_token:
            If True, append [SEG] to training answers when missing.
        merge_masks_for_single_seg:
            If True and answer contains <=1 [SEG], merge multiple GT masks into one union.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor,
        image_size: int = 1024,
        max_seq_length: int = 2048,
        split: str = "train",
        seg_token: str = "[SEG]",
        seg_token_idx: Optional[int] = None,
        force_seg_token: bool = True,
        merge_masks_for_single_seg: bool = True,
        image_key: str = "image",
        query_key: str = "query",
        answer_key: str = "outputs",
        mask_key: str = "json",
        **kwargs,
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_size = int(image_size)
        self.max_seq_length = int(max_seq_length)
        self.split = split

        self.seg_token = seg_token
        self.seg_token_idx = seg_token_idx
        self.force_seg_token = bool(force_seg_token)
        self.merge_masks_for_single_seg = bool(merge_masks_for_single_seg)

        self.image_key = image_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.mask_key = mask_key

        self.kwargs = kwargs
        self.transform = ResizeLongestSide(self.image_size)

        self.samples = self._load_samples()

        logging.info(
            f"Loaded {len(self.samples)} ReaSeg reason-seg samples "
            f"from {data_path}, split={split}."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        try:
            return self._process_sample(sample)
        except Exception:
            logging.exception(f"Failed to process sample idx={idx}: {sample}")
            raise

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_samples(self) -> List[Dict[str, Any]]:
        if os.path.isfile(self.data_path):
            json_file = self.data_path
        else:
            json_file = os.path.join(self.data_path, f"{self.split}.json")

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Dataset JSON not found: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            raise ValueError(f"Dataset JSON should be a list or dict, got {type(data)}")

        return data

    # ------------------------------------------------------------------
    # Path / image helpers
    # ------------------------------------------------------------------

    def _resolve_path(
        self,
        path_value: Union[str, os.PathLike, None],
        required: bool = True,
        desc: str = "file",
        extra_dirs: Optional[List[str]] = None,
    ) -> str:
        """
        Resolve paths under common layouts:
            1. absolute path
            2. data_path/path
            3. data_path/basename(path)
            4. data_path/images/path
            5. data_path/masks/path
            6. data_path/reason_seg/ReasonSeg/train/path
            7. dirname(data_path)/path
        """
        if path_value is None or str(path_value).strip() == "":
            if required:
                raise FileNotFoundError(f"Empty {desc} path.")
            return ""

        path_value = str(path_value)
        candidates: List[str] = []

        if os.path.isabs(path_value):
            candidates.append(path_value)
        else:
            base_dir = self.data_path if os.path.isdir(self.data_path) else os.path.dirname(self.data_path)
            parent_dir = os.path.dirname(base_dir)

            candidates.extend(
                [
                    os.path.join(base_dir, path_value),
                    os.path.join(base_dir, os.path.basename(path_value)),
                    os.path.join(base_dir, "images", path_value),
                    os.path.join(base_dir, "images", os.path.basename(path_value)),
                    os.path.join(base_dir, "masks", path_value),
                    os.path.join(base_dir, "masks", os.path.basename(path_value)),
                    os.path.join(base_dir, "json", path_value),
                    os.path.join(base_dir, "json", os.path.basename(path_value)),
                    os.path.join(base_dir, "reason_seg", "ReasonSeg", "train", path_value),
                    os.path.join(base_dir, "reason_seg", "ReasonSeg", "train", os.path.basename(path_value)),
                    os.path.join(parent_dir, path_value),
                    os.path.join(parent_dir, os.path.basename(path_value)),
                ]
            )

            if extra_dirs:
                for d in extra_dirs:
                    candidates.append(os.path.join(base_dir, d, path_value))
                    candidates.append(os.path.join(base_dir, d, os.path.basename(path_value)))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        if required:
            raise FileNotFoundError(
                f"Cannot resolve {desc} path: {path_value}. Tried: {candidates}"
            )

        return ""

    def _load_image(self, image_path: str) -> Tuple[np.ndarray, Image.Image, str]:
        full_image_path = self._resolve_path(
            image_path,
            required=True,
            desc="image",
            extra_dirs=["images", "image"],
        )

        image_bgr = cv2.imread(full_image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {full_image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        return image_rgb, pil_image, full_image_path

    def _build_medsam_image(self, image_rgb: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Build MedSAM image tensor.

        Returns:
            medsam_image:
                Tensor[3, image_size, image_size], normalized and padded.
            resize_shape:
                (resized_h, resized_w), before padding.
        """
        image_resized = self.transform.apply_image(image_rgb)
        resize_shape = tuple(int(x) for x in image_resized.shape[:2])

        medsam_image = self._preprocess_image_for_medsam(image_resized)

        return medsam_image, resize_shape

    def _preprocess_image_for_medsam(self, image: np.ndarray) -> torch.Tensor:
        """
        Normalize and pad RGB image for MedSAM/SAM image encoder.

        Args:
            image:
                Resized RGB image, HWC, uint8 or float-like 0-255.

        Returns:
            Tensor[3, image_size, image_size].
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray image, got {type(image)}")

        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        image_tensor = (image_tensor - pixel_mean) / pixel_std

        _, h, w = image_tensor.shape
        pad_h = self.image_size - h
        pad_w = self.image_size - w

        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Resized MedSAM image is larger than image_size={self.image_size}: "
                f"got {(h, w)}"
            )

        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))

        return image_tensor

    # ------------------------------------------------------------------
    # Qwen2.5-VL input / label helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, conversation, add_generation_prompt: bool) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        if hasattr(self.processor, "tokenizer") and hasattr(
            self.processor.tokenizer,
            "apply_chat_template",
        ):
            return self.processor.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        raise RuntimeError(
            "No apply_chat_template found in processor, processor.tokenizer, or tokenizer."
        )

    def _check_image_tokens(self, inputs: Dict[str, torch.Tensor], image_path: str = "") -> None:
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        if image_token_id is None or image_token_id < 0:
            logging.warning("Cannot find <|image_pad|> token id; skip image-token check.")
            return

        num_image_tokens = (inputs["input_ids"] == image_token_id).sum().item()

        if num_image_tokens == 0:
            decoded = self.tokenizer.decode(
                inputs["input_ids"][0],
                skip_special_tokens=False,
            )

            raise RuntimeError(
                "Qwen2.5-VL image-token mismatch: input_ids has 0 <|image_pad|> "
                "tokens, but image was provided.\n"
                f"image_path: {image_path}\n"
                f"decoded preview:\n{decoded[:1200]}"
            )

    def _build_inputs_and_labels(
        self,
        query: str,
        answer: str,
        pil_image: Image.Image,
        compute_text_loss: bool = True,
        image_path: str = "",
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Build Qwen2.5-VL inputs and assistant-only labels.

        Train split:
            user + assistant answer, labels only assistant answer tokens.

        Val/test split:
            user + generation prompt, labels all -100.
        """
        query = "" if query is None else str(query)
        answer = "" if answer is None else str(answer)

        if self.split == "train":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ],
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]

            text = self._apply_chat_template(
                conversation,
                add_generation_prompt=False,
            )
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            text = self._apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )

        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=False,
        )

        self._check_image_tokens(inputs, image_path=image_path)

        labels = inputs["input_ids"].clone()

        if self.split == "train" and compute_text_loss:
            prompt_conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            prompt_text = self._apply_chat_template(
                prompt_conversation,
                add_generation_prompt=True,
            )

            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[pil_image],
                return_tensors="pt",
                padding=False,
            )

            prompt_token_len = prompt_inputs["input_ids"].shape[1]
            labels[0, :prompt_token_len] = -100

            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels[:] = -100

        if "image_grid_thw" not in inputs or inputs["image_grid_thw"] is None:
            raise RuntimeError(
                "Qwen2.5-VL processor did not return image_grid_thw. "
                "Please check processor/model compatibility."
            )

        return inputs, labels

    # ------------------------------------------------------------------
    # Mask loading / parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_masks(image_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = image_shape
        return torch.zeros((1, h, w), dtype=torch.float32)

    def _ensure_masks_3d(
        self,
        masks: Union[torch.Tensor, np.ndarray],
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)

        masks = masks.float()

        if masks.dim() == 2:
            masks = masks.unsqueeze(0)

        if masks.dim() != 3:
            raise ValueError(f"Expected masks [N,H,W] or [H,W], got {tuple(masks.shape)}")

        h, w = image_shape

        if tuple(masks.shape[-2:]) != (h, w):
            resized = []
            for mask in masks:
                mask_np = mask.detach().cpu().numpy().astype(np.float32)
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                resized.append(torch.from_numpy(mask_np))
            masks = torch.stack(resized, dim=0).float()

        masks = (masks > 0).float()

        return masks

    def _load_masks_from_any(
        self,
        mask_source: Any,
        image_shape: Tuple[int, int],
        required: bool = False,
    ) -> torch.Tensor:
        if mask_source is None or mask_source == "":
            if required:
                raise FileNotFoundError("Mask source is required but empty.")
            return self._empty_masks(image_shape)

        if isinstance(mask_source, torch.Tensor):
            return self._ensure_masks_3d(mask_source, image_shape)

        if isinstance(mask_source, np.ndarray):
            return self._ensure_masks_3d(mask_source, image_shape)

        if isinstance(mask_source, str):
            mask_path = self._resolve_path(
                mask_source,
                required=required,
                desc="mask",
                extra_dirs=["masks", "mask", "json"],
            )

            if mask_path == "":
                return self._empty_masks(image_shape)

            lower_path = mask_path.lower()

            if lower_path.endswith(".json"):
                return self._load_masks_from_json(mask_path, image_shape)

            if lower_path.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is None:
                    if required:
                        raise FileNotFoundError(f"Cannot load mask image: {mask_path}")
                    logging.warning(f"Cannot load mask image: {mask_path}; using empty mask.")
                    return self._empty_masks(image_shape)

                return self._ensure_masks_3d(mask, image_shape)

            raise ValueError(f"Unsupported mask file type: {mask_path}")

        if isinstance(mask_source, dict):
            if "shapes" in mask_source and isinstance(mask_source["shapes"], list):
                shape_masks = []
                for shape in mask_source["shapes"]:
                    shape_masks.append(self._parse_single_mask(shape, image_shape))

                if len(shape_masks) == 0:
                    return self._empty_masks(image_shape)

                return self._ensure_masks_3d(torch.stack(shape_masks, dim=0), image_shape)

            for list_key in ["annotations", "objects", "masks"]:
                if list_key in mask_source and isinstance(mask_source[list_key], list):
                    return self._load_masks_from_any(mask_source[list_key], image_shape)

            mask = self._parse_single_mask(mask_source, image_shape)
            return self._ensure_masks_3d(mask, image_shape)

        if isinstance(mask_source, list):
            if len(mask_source) == 0:
                return self._empty_masks(image_shape)

            # Dense numeric mask list.
            try:
                numeric_arr = np.array(mask_source, dtype=np.float32)
                if numeric_arr.ndim in (2, 3) and numeric_arr.shape[-2:] == image_shape:
                    return self._ensure_masks_3d(numeric_arr, image_shape)
            except Exception:
                pass

            # Single polygon.
            if self._looks_like_polygon(mask_source):
                mask = self._polygon_to_mask(mask_source, image_shape)
                return self._ensure_masks_3d(mask, image_shape)

            # List of annotations.
            parsed_masks = [
                self._parse_single_mask(mask_info, image_shape)
                for mask_info in mask_source
            ]

            return self._ensure_masks_3d(torch.stack(parsed_masks, dim=0), image_shape)

        logging.warning(f"Unsupported mask source type: {type(mask_source)}; using empty mask.")
        return self._empty_masks(image_shape)

    def _load_masks_from_json(self, json_path: str, image_shape: Tuple[int, int]) -> torch.Tensor:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                mask_data = json.load(f)
            return self._load_masks_from_any(mask_data, image_shape, required=False)
        except Exception as e:
            logging.warning(f"Failed to load masks from {json_path}: {e}")
            return self._empty_masks(image_shape)

    def _parse_single_mask(self, mask_info: Any, image_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = image_shape

        if isinstance(mask_info, torch.Tensor):
            return self._ensure_masks_3d(mask_info, image_shape)[0]

        if isinstance(mask_info, np.ndarray):
            return self._ensure_masks_3d(mask_info, image_shape)[0]

        if isinstance(mask_info, str):
            return self._load_masks_from_any(mask_info, image_shape, required=False)[0]

        if isinstance(mask_info, list):
            if self._looks_like_polygon(mask_info):
                return self._polygon_to_mask(mask_info, image_shape)

            try:
                arr = np.array(mask_info, dtype=np.float32)
                if arr.ndim in (2, 3):
                    return self._ensure_masks_3d(arr, image_shape)[0]
            except Exception:
                pass

        if isinstance(mask_info, dict):
            if "mask" in mask_info:
                return self._ensure_masks_3d(np.array(mask_info["mask"]), image_shape)[0]

            if "polygon" in mask_info:
                return self._polygon_to_mask(mask_info["polygon"], image_shape)

            if "points" in mask_info:
                return self._polygon_to_mask(mask_info["points"], image_shape)

            if "segmentation" in mask_info:
                return self._parse_single_mask(mask_info["segmentation"], image_shape)

            if "bbox" in mask_info:
                return self._bbox_to_mask(mask_info["bbox"], image_shape)

        return torch.zeros((h, w), dtype=torch.float32)

    @staticmethod
    def _looks_like_polygon(obj: Any) -> bool:
        if not isinstance(obj, list) or len(obj) == 0:
            return False

        # Flat polygon: [x1, y1, x2, y2, ...]
        if all(isinstance(x, (int, float)) for x in obj):
            return len(obj) >= 6 and len(obj) % 2 == 0

        # Point list: [[x, y], [x, y], ...]
        if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in obj):
            first = obj[0]
            if len(first) >= 2 and all(isinstance(v, (int, float)) for v in first[:2]):
                return len(obj) >= 3

        return False

    @staticmethod
    def _normalize_polygon_points(poly: Any) -> List[Tuple[float, float]]:
        # Flat polygon: [x1, y1, x2, y2, ...]
        if isinstance(poly, list) and all(isinstance(x, (int, float)) for x in poly):
            return [
                (float(poly[i]), float(poly[i + 1]))
                for i in range(0, len(poly), 2)
            ]

        points = []
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append((float(p[0]), float(p[1])))

        return points

    def _polygon_to_mask(self, polygon: Any, image_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = image_shape
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        if polygon is None:
            return torch.zeros((h, w), dtype=torch.float32)

        # Multiple polygons: [[[x,y],...], [[x,y],...]]
        is_multiple = (
            isinstance(polygon, list)
            and len(polygon) > 0
            and isinstance(polygon[0], list)
            and len(polygon[0]) > 0
            and isinstance(polygon[0][0], (list, tuple))
        )

        polygons = polygon if is_multiple else [polygon]

        for poly in polygons:
            points = self._normalize_polygon_points(poly)
            if len(points) >= 3:
                draw.polygon(points, outline=1, fill=1)

        return torch.from_numpy(np.array(mask, dtype=np.float32))

    @staticmethod
    def _bbox_to_mask(bbox: List[float], image_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = image_shape
        mask = torch.zeros((h, w), dtype=torch.float32)

        if bbox is None or len(bbox) != 4:
            return mask

        x1, y1, x2, y2 = [float(v) for v in bbox]

        # Assume [x1, y1, x2, y2]. If invalid, fall back to COCO [x, y, w, h].
        if x2 <= x1 or y2 <= y1:
            x2 = x1 + max(0.0, x2)
            y2 = y1 + max(0.0, y2)

        x1 = max(0, min(w, int(round(x1))))
        y1 = max(0, min(h, int(round(y1))))
        x2 = max(0, min(w, int(round(x2))))
        y2 = max(0, min(h, int(round(y2))))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Sample processing
    # ------------------------------------------------------------------

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_path = (
            sample.get(self.image_key)
            or sample.get("image")
            or sample.get("image_path")
            or ""
        )

        query = (
            sample.get(self.query_key)
            or sample.get("query")
            or sample.get("input_text")
            or sample.get("question")
            or sample.get("input")
            or ""
        )

        answer = (
            sample.get(self.answer_key)
            or sample.get("outputs")
            or sample.get("answer")
            or sample.get("output")
            or ""
        )

        mask_source = (
            sample.get(self.mask_key)
            or sample.get("json")
            or sample.get("mask")
            or sample.get("mask_path")
            or None
        )

        image_rgb, pil_image, full_image_path = self._load_image(image_path)
        original_size = tuple(int(x) for x in image_rgb.shape[:2])

        medsam_image, resize_shape = self._build_medsam_image(image_rgb)

        gt_masks = self._load_masks_from_any(
            mask_source,
            image_shape=original_size,
            required=False,
        )

        answer = "" if answer is None else str(answer)

        if self.split == "train" and self.force_seg_token and "[SEG]" not in answer:
            answer = answer.rstrip() + " [SEG]"

        if self.merge_masks_for_single_seg and isinstance(gt_masks, torch.Tensor):
            seg_count = answer.count("[SEG]")
            if seg_count <= 1 and gt_masks.dim() == 3 and gt_masks.shape[0] > 1:
                gt_masks = gt_masks.max(dim=0, keepdim=True).values.float()

        inputs, labels = self._build_inputs_and_labels(
            query=query,
            answer=answer,
            pil_image=pil_image,
            compute_text_loss=True,
            image_path=full_image_path,
        )

        item = {
            # Qwen2.5-VL branch
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),

            # Preferred ReaSeg names
            "medsam_image": medsam_image,
            "gt_masks": gt_masks,
            "original_size": original_size,
            "resize_shape": resize_shape,
            "image_path": full_image_path,

            # Useful metadata
            "query": query,
            "answer": answer,
        }

        # Backward-compatible aliases for older code paths.
        item["images"] = item["medsam_image"]
        item["masks_list"] = item["gt_masks"]
        item["original_size_list"] = item["original_size"]
        item["resize_list"] = [item["resize_shape"]]

        return item


# Backward-compatible aliases
ReasonSegDataset = ReaSegReasonSegDataset
FinetuneDataset = ReaSegReasonSegDataset


__all__ = [
    "ReaSegReasonSegDataset",
    "ReasonSegDataset",
    "FinetuneDataset",
]