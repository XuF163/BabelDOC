import ast
import logging
import os
import platform
import re
import threading
from collections.abc import Generator

import cv2
import numpy as np

from babeldoc.docvision.base_doclayout import DocLayoutModel
from babeldoc.docvision.base_doclayout import YoloResult
from babeldoc.format.pdf.document_il.utils.mupdf_helper import get_no_rotation_img

try:
    import onnx
    import onnxruntime
except ImportError as e:
    if "DLL load failed" in str(e):
        raise OSError(
            "Microsoft Visual C++ Redistributable is not installed. "
            "Download it at https://aka.ms/vs/17/release/vc_redist.x64.exe"
        ) from e
    raise
import pymupdf

import babeldoc.format.pdf.document_il.il_version_1
from babeldoc.assets.assets import get_doclayout_onnx_model_path

# from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def _parse_int_env(name: str) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_file(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_available_memory_bytes() -> int | None:
    """
    Best-effort available memory detection (container-aware).

    Priority:
    1) cgroup v2 (memory.max - memory.current)
    2) cgroup v1 (memory.limit_in_bytes - memory.usage_in_bytes)
    3) /proc/meminfo MemAvailable
    """
    try:
        limit = _read_int_file("/sys/fs/cgroup/memory.max")
        current = _read_int_file("/sys/fs/cgroup/memory.current")
        if limit is not None and current is not None:
            if 0 < limit < (1 << 60) and current >= 0:
                avail = limit - current
                if avail > 0:
                    return avail
    except Exception:
        pass

    try:
        limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        current = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if limit is not None and current is not None:
            if 0 < limit < (1 << 60) and current >= 0:
                avail = limit - current
                if avail > 0:
                    return avail
    except Exception:
        pass

    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    break
                return int(parts[1]) * 1024
    except Exception:
        pass

    return None


def _resolve_doclayout_batch_size(total_images: int, requested_batch_size: int | None = None) -> int:
    """
    Resolve DocLayout batch size.

    Priority:
    1) BABELDOC_DOCLAYOUT_BATCH_SIZE (absolute override)
    2) requested_batch_size (caller hint)
    3) dynamic default (more aggressive for throughput)

    Additional caps:
    - BABELDOC_DOCLAYOUT_MAX_BATCH_SIZE / BABELDOC_DOCLAYOUT_MAX_BATCH (default 64)
    - available memory cap (best-effort):
        budget = MemAvailable * BABELDOC_DOCLAYOUT_AUTO_BATCH_MEM_FRACTION (default 0.25)
        per_image_mb = BABELDOC_DOCLAYOUT_PER_IMAGE_MB (default 24)
    """
    if total_images <= 0:
        return 0

    env_bs = _parse_int_env("BABELDOC_DOCLAYOUT_BATCH_SIZE")
    if env_bs > 0:
        return min(env_bs, total_images)

    resolved = 0
    if requested_batch_size is not None:
        try:
            resolved = int(requested_batch_size)
        except (TypeError, ValueError):
            resolved = 0

    if resolved <= 0:
        if total_images > 64:
            resolved = 64
        elif total_images > 32:
            resolved = 32
        elif total_images > 8:
            resolved = 16
        else:
            resolved = 4

    max_bs = (
        _parse_int_env("BABELDOC_DOCLAYOUT_MAX_BATCH_SIZE")
        or _parse_int_env("BABELDOC_DOCLAYOUT_MAX_BATCH")
    )
    if max_bs <= 0:
        max_bs = 64

    mem_fraction = _parse_float_env("BABELDOC_DOCLAYOUT_AUTO_BATCH_MEM_FRACTION", 0.25)
    mem_fraction = max(0.0, min(mem_fraction, 1.0))
    per_image_mb = _parse_int_env("BABELDOC_DOCLAYOUT_PER_IMAGE_MB")
    if per_image_mb <= 0:
        per_image_mb = 24

    mem_avail = _get_available_memory_bytes()
    if mem_avail is not None and mem_fraction > 0 and per_image_mb > 0:
        budget_mb = int(mem_avail * mem_fraction / (1024 * 1024))
        if budget_mb > 0:
            mem_cap = max(1, budget_mb // per_image_mb)
            resolved = min(resolved, mem_cap)

    resolved = min(resolved, max_bs, total_images)
    return max(1, resolved)


# 检测操作系统类型
os_name = platform.system()


class OnnxModel(DocLayoutModel):
    def __init__(self, model_path: str):
        self.model_path = model_path

        model = onnx.load(model_path)
        metadata = {d.key: d.value for d in model.metadata_props}
        self._stride = ast.literal_eval(metadata["stride"])
        self._names = ast.literal_eval(metadata["names"])
        providers: list[str] = []

        available_providers = onnxruntime.get_available_providers()

        # Default to CPU-only for stability. Users can explicitly opt-in to other
        # providers (CUDA/DirectML/etc) for speed via env vars.
        env_providers = os.getenv("BABELDOC_DOCLAYOUT_PROVIDERS", "").strip()
        allow_non_cpu = os.getenv("BABELDOC_DOCLAYOUT_ALLOW_NON_CPU", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if env_providers:
            wanted = [p.strip() for p in env_providers.split(",") if p.strip()]
            providers = [p for p in wanted if p in available_providers]
            if not providers:
                logger.warning(
                    "BABELDOC_DOCLAYOUT_PROVIDERS is set but none matched available providers: %s",
                    available_providers,
                )
        if not providers:
            if allow_non_cpu:
                providers = list(available_providers)
            else:
                for provider in available_providers:
                    if re.match(r"cpu", provider, re.IGNORECASE):
                        providers.append(provider)
        if not providers:
            # Last-resort: let onnxruntime decide.
            providers = list(available_providers)
        if providers:
            logger.info("DocLayout ONNX providers: %s", providers)
        self.model = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=providers,
        )
        self.lock = threading.Lock()

    @staticmethod
    def from_pretrained():
        pth = get_doclayout_onnx_model_path()
        return OnnxModel(pth)

    @property
    def stride(self):
        return self._stride

    def resize_and_pad_image(self, image, new_shape):
        """
        Resize and pad the image to the specified size, ensuring dimensions are multiples of stride.

        Parameters:
        - image: Input image
        - new_shape: Target size (integer or (height, width) tuple)
        - stride: Padding alignment stride, default 32

        Returns:
        - Processed image
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = image.shape[:2]
        new_h, new_w = new_shape

        # Calculate scaling ratio
        r = min(new_h / h, new_w / w)
        resized_h, resized_w = int(round(h * r)), int(round(w * r))

        # Resize image
        image = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Calculate padding size and align to stride multiple
        pad_w = (new_w - resized_w) % self.stride
        pad_h = (new_h - resized_h) % self.stride
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        # Add padding
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return image

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for,
                in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """

        # Calculate scaling ratio
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

        # Calculate padding size
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)

        # Remove padding and scale boxes
        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes

    def predict(self, image, imgsz=800, batch_size=16, **kwargs):
        """
        Predict the layout of document pages.

        Args:
            image: A single image or a list of images of document pages.
            imgsz: Resize the image to this size. Must be a multiple of the stride.
            batch_size: Number of images to process in one batch.
            **kwargs: Additional arguments.

        Returns:
            A list of YoloResult objects, one for each input image.
        """
        # Handle single image input
        if isinstance(image, np.ndarray) and len(image.shape) == 3:
            image = [image]

        total_images = len(image)
        results = []

        batch_size = _resolve_doclayout_batch_size(total_images, batch_size)

        # Process images in batches
        for i in range(0, total_images, batch_size):
            batch_images = image[i : i + batch_size]
            batch_size_actual = len(batch_images)

            # Calculate target size based on the maximum height in the batch
            target_imgsz = 1024

            # Preprocess batch
            processed_batch = []
            orig_shapes = []
            for img in batch_images:
                orig_h, orig_w = img.shape[:2]
                orig_shapes.append((orig_h, orig_w))

                pix = self.resize_and_pad_image(img, new_shape=target_imgsz)
                pix = np.transpose(pix, (2, 0, 1))  # CHW
                pix = pix.astype(np.float32) / 255.0  # Normalize to [0, 1]
                processed_batch.append(pix)

            # Stack batch
            batch_input = np.stack(processed_batch, axis=0)  # BCHW
            new_h, new_w = batch_input.shape[2:]

            # Run inference
            batch_preds = self.model.run(None, {"images": batch_input})[0]

            # Process each prediction in the batch
            for j in range(batch_size_actual):
                preds = batch_preds[j]
                preds = preds[preds[..., 4] > 0.25]
                if len(preds) > 0:
                    preds[..., :4] = self.scale_boxes(
                        (new_h, new_w),
                        preds[..., :4],
                        orig_shapes[j],
                    )
                results.append(YoloResult(boxes_data=preds, names=self._names))

        return results

    def handle_document(
        self,
        pages: list[babeldoc.format.pdf.document_il.il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ) -> Generator[
        tuple[babeldoc.format.pdf.document_il.il_version_1.Page, YoloResult], None, None
    ]:
        # Keep memory bounded by batching page images, while still reducing
        # per-page ONNX invocation overhead.
        total_pages = len(pages)
        handle_batch_size = _resolve_doclayout_batch_size(total_pages, None)
        logger.info(
            "DocLayout batch size resolved to %s (pages=%s)",
            handle_batch_size,
            total_pages,
        )

        batch_pages: list[babeldoc.format.pdf.document_il.il_version_1.Page] = []
        batch_images: list[np.ndarray] = []

        def flush_batch():
            if not batch_pages:
                return
            preds = self.predict(batch_images, batch_size=handle_batch_size)
            for page, image, pred in zip(batch_pages, batch_images, preds, strict=True):
                save_debug_image(image, pred, page.page_number + 1)
                yield page, pred

        for page in pages:
            translate_config.raise_if_cancelled()
            with self.lock:
                pix = get_no_rotation_img(mupdf_doc[page.page_number])
            image = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height,
                pix.width,
                3,
            )[:, :, ::-1]

            batch_pages.append(page)
            batch_images.append(image)
            if len(batch_pages) >= handle_batch_size:
                yield from flush_batch()
                batch_pages.clear()
                batch_images.clear()

        if batch_pages:
            yield from flush_batch()
