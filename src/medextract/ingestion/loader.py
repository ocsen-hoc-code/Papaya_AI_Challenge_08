"""Load medical documents (PDF or raster image) as normalized RGB page images.

The VLM only sees images, so every input is turned into a list of RGB pages whose long
side is capped at ``max_side``. PDFs are rendered with PyMuPDF; if they carry an
embedded text layer, that text is returned too so downstream code can cross-check it.

The file type is detected from magic bytes, never from the extension.

Note: the module is imported as ``pymupdf`` (the same library as ``fitz``). PyMuPDF 2.x
emits a deprecation warning for the ``fitz`` alias.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

PAGE_SEPARATOR = "\n\f\n"
DEFAULT_MAX_SIDE = 1600
DEFAULT_DPI = 200
DEFAULT_MAX_PAGES = 10

_LANCZOS = Image.Resampling.LANCZOS


class IngestError(Exception):
    """The input is missing, unreadable, corrupt, encrypted or of an unsupported type."""


@dataclass
class DocumentInput:
    path: Path
    doc_id: str  # file stem, e.g. "01_receipt_en"
    mime: str  # "application/pdf", "image/png", "image/jpeg", ...
    images: list[Image.Image]  # RGB, one per page, long side <= max_side
    text_layer: str | None  # embedded PDF text, pages joined by "\n\f\n"; None if empty
    page_count: int  # true page count, which may exceed len(images)


def detect_mime(data: bytes) -> str:
    """Return the MIME type of ``data`` from its leading magic bytes.

    Raises IngestError for empty or unsupported content.
    """
    if not data:
        raise IngestError("File is empty")
    if b"%PDF-" in data[:1024]:
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    raise IngestError("Unsupported file type: expected PDF, PNG, JPEG, WEBP or TIFF content")


def load_document(
    path: str | Path,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    dpi: int = DEFAULT_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> DocumentInput:
    """Load a PDF or image file into normalized RGB page images.

    Args:
        path: File to load. The type is detected from its content.
        max_side: Maximum length of the longer side of each returned image, in pixels.
        dpi: Resolution used to render PDF pages.
        max_pages: Render at most this many pages (PDF pages or TIFF frames).

    Raises:
        IngestError: The file is missing, unreadable, corrupt, encrypted or unsupported.
        ValueError: One of the numeric options is not positive.
    """
    if max_side < 1 or dpi < 1 or max_pages < 1:
        raise ValueError("max_side, dpi and max_pages must be >= 1")

    file_path = Path(path)
    data = _read_bytes(file_path)
    mime = detect_mime(data)

    if mime == "application/pdf":
        images, text_layer, page_count = _load_pdf(data, file_path, dpi=dpi, max_side=max_side, max_pages=max_pages)
    else:
        images, page_count = _load_raster(data, file_path, max_side=max_side, max_pages=max_pages, mime=mime)
        text_layer = None

    return DocumentInput(
        path=file_path,
        doc_id=file_path.stem,
        mime=mime,
        images=images,
        text_layer=text_layer,
        page_count=page_count,
    )


async def aload_document(path: str | Path, **kwargs) -> DocumentInput:
    """Async wrapper around ``load_document`` that runs it in a worker thread."""
    return await asyncio.to_thread(load_document, path, **kwargs)


# --- Helpers ---------------------------------------------------------------------------


def _read_bytes(file_path: Path) -> bytes:
    if not file_path.exists():
        raise IngestError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise IngestError(f"Not a regular file: {file_path}")
    try:
        return file_path.read_bytes()
    except OSError as exc:
        raise IngestError(f"Cannot read {file_path}: {exc}") from exc


def _fit(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the long side is at most ``max_side``. Never upscales."""
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, _LANCZOS)


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert to RGB, compositing any transparency onto a white background."""
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _normalize_raster(image: Image.Image, max_side: int) -> Image.Image:
    """Apply EXIF orientation, convert to RGB and downscale one decoded frame."""
    image = ImageOps.exif_transpose(image)
    image = _to_rgb(image)
    return _fit(image, max_side)


def _load_raster(
    data: bytes, file_path: Path, *, max_side: int, max_pages: int, mime: str
) -> tuple[list[Image.Image], int]:
    try:
        opened = Image.open(BytesIO(data))
        opened.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise IngestError(f"Corrupt or unreadable image {file_path.name}: {exc}") from exc

    with opened:
        page_count = getattr(opened, "n_frames", 1)
        # Only multi-page TIFFs are treated as several pages. Other formats use the first frame.
        if mime == "image/tiff" and page_count > 1:
            frames = islice(ImageSequence.Iterator(opened), max_pages)
            images = [_normalize_raster(frame.copy(), max_side) for frame in frames]
        else:
            images = [_normalize_raster(opened.copy(), max_side)]
    return images, page_count


def _load_pdf(
    data: bytes, file_path: Path, *, dpi: int, max_side: int, max_pages: int
) -> tuple[list[Image.Image], str | None, int]:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for corrupt input
        raise IngestError(f"Corrupt or unreadable PDF {file_path.name}: {exc}") from exc

    with document:
        if document.needs_pass and not document.authenticate(""):
            raise IngestError(f"Encrypted PDF requires a password: {file_path.name}")
        page_count = document.page_count
        if page_count == 0:
            raise IngestError(f"PDF has no pages: {file_path.name}")

        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        images: list[Image.Image] = []
        texts: list[str] = []
        try:
            for index in range(min(page_count, max_pages)):
                page = document[index]
                pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
                images.append(_fit(pix.pil_image().convert("RGB"), max_side))
                texts.append(page.get_text())
        except Exception as exc:
            raise IngestError(f"Cannot render PDF {file_path.name}: {exc}") from exc

    joined = PAGE_SEPARATOR.join(texts)
    text_layer = joined if joined.strip() else None
    return images, text_layer, page_count
