import math
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

# keep architecture constants used by infer/qwen utils
ARCHITECTURE_QWEN_IMAGE = "qi"
ARCHITECTURE_QWEN_IMAGE_FULL = "qwen_image"
ARCHITECTURE_QWEN_IMAGE_EDIT = "qie"
ARCHITECTURE_QWEN_IMAGE_EDIT_FULL = "qwen_image_edit"


# architecture -> bucket step
ARCHITECTURE_STEPS_MAP = {
    ARCHITECTURE_QWEN_IMAGE: 16,
    ARCHITECTURE_QWEN_IMAGE_EDIT: 16,
}


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize + center-crop image to bucket resolution.

    bucket_reso is (width, height).
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil else image

    bucket_width, bucket_height = bucket_reso
    scale = max(bucket_width / image_width, bucket_height / image_height)

    resized_w = int(image_width * scale + 0.5)
    resized_h = int(image_height * scale + 0.5)

    pil_image = image if is_pil else Image.fromarray(image)
    # LANCZOS works for both upsample/downsample and avoids cv2 dependency.
    pil_image = pil_image.resize((resized_w, resized_h), Image.LANCZOS)
    image_np = np.array(pil_image)

    crop_left = max(0, (resized_w - bucket_width) // 2)
    crop_top = max(0, (resized_h - bucket_height) // 2)
    image_np = image_np[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]

    return image_np


class BucketSelector:
    """Minimal bucket selector used by qwen_image_utils for control image preprocessing."""

    @classmethod
    def calculate_bucket_resolution(
        cls,
        image_size: tuple[int, int],
        resolution: tuple[int, int],
        reso_steps: Optional[int] = None,
        architecture: Optional[str] = None,
    ) -> tuple[int, int]:
        if reso_steps is None:
            if architecture is None:
                raise ValueError("resolution steps or architecture must be provided")
            if architecture not in ARCHITECTURE_STEPS_MAP:
                raise ValueError(f"Invalid architecture: {architecture}")
            reso_steps = ARCHITECTURE_STEPS_MAP[architecture]

        max_area = resolution[0] * resolution[1]
        width, height = image_size
        aspect_ratio = width / height

        bucket_width = int(math.sqrt(max_area * aspect_ratio))
        bucket_height = int(math.sqrt(max_area / aspect_ratio))
        bucket_width = divisible_by(bucket_width, reso_steps)
        bucket_height = divisible_by(bucket_height, reso_steps)

        best_resolution = None
        best_diff = float("inf")
        for i in range(-2, 3):
            w = bucket_width + i * reso_steps
            if w <= 0:
                continue
            h = divisible_by(max_area // w, reso_steps)
            if h <= 0:
                continue
            diff = abs((w / h) - aspect_ratio)
            if diff < best_diff:
                best_diff = diff
                best_resolution = (w, h)

        if best_resolution is not None:
            return best_resolution
        return bucket_width, bucket_height
