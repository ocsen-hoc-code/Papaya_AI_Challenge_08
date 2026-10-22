from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from medextract.ingestion.loader import (
    PAGE_SEPARATOR,
    DocumentInput,
    IngestError,
    aload_document,
    detect_mime,
    load_document,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCUMENT_TESTS_DIR = REPO_ROOT / "document-tests"


# --- Helpers ---------------------------------------------------------------------------


def make_pdf(path: Path, pages: list[str | None], size: tuple[float, float] = (612, 792)) -> Path:
    """Write a PDF with one page per entry. ``None`` makes a page with no text layer."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=size[0], height=size[1])
        if text is not None:
            page.insert_text((72, 100), text, fontsize=14)
        else:
            page.draw_rect(pymupdf.Rect(72, 72, 300, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    doc.save(path)
    doc.close()
    return path


def make_image(path: Path, size: tuple[int, int], mode: str = "RGB", color=(200, 30, 30), fmt=None) -> Path:
    Image.new(mode, size, color).save(path, format=fmt)
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


# --- detect_mime -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"%PDF-1.7\n rest", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 8, "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"II*\x00\x08\x00\x00\x00", "image/tiff"),
        (b"MM\x00*\x00\x00\x00\x08", "image/tiff"),
    ],
)
def test_detect_mime_by_magic_bytes(data, expected):
    assert detect_mime(data) == expected


def test_detect_mime_rejects_unsupported_and_empty_content():
    with pytest.raises(IngestError, match="Unsupported"):
        detect_mime(b"GIF89a....")
    with pytest.raises(IngestError, match="empty"):
        detect_mime(b"")


def test_wrong_extension_is_ignored(tmp_path):
    png = make_image(tmp_path / "real.png", (50, 40), fmt="PNG")
    png_as_pdf = write_bytes(tmp_path / "scan.pdf", png.read_bytes())

    loaded = load_document(png_as_pdf)
    assert loaded.mime == "image/png"
    assert loaded.text_layer is None
    assert loaded.images[0].size == (50, 40)

    pdf_as_png = make_pdf(tmp_path / "doc.png", ["Hello"])
    loaded = load_document(pdf_as_png)
    assert loaded.mime == "application/pdf"
    assert loaded.text_layer is not None


def test_jpeg_saved_with_png_extension_loads_as_jpeg(tmp_path):
    path = make_image(tmp_path / "photo.png", (30, 20), fmt="JPEG")
    loaded = load_document(path)
    assert loaded.mime == "image/jpeg"
    assert loaded.images[0].mode == "RGB"


# --- PDF -------------------------------------------------------------------------------


def test_pdf_single_page_extracts_text_and_renders_rgb(tmp_path):
    path = make_pdf(tmp_path / "01_receipt_en.pdf", ["Total 1500 USD"])

    loaded = load_document(path)

    assert isinstance(loaded, DocumentInput)
    assert loaded.path == path
    assert loaded.doc_id == "01_receipt_en"
    assert loaded.mime == "application/pdf"
    assert loaded.page_count == 1
    assert len(loaded.images) == 1
    assert loaded.images[0].mode == "RGB"
    assert "Total 1500 USD" in loaded.text_layer


def test_pdf_multi_page_text_is_joined_with_separator(tmp_path):
    path = make_pdf(tmp_path / "multi.pdf", ["First page", "Second page", "Third page"])

    loaded = load_document(path)

    assert loaded.page_count == 3
    assert len(loaded.images) == 3
    parts = loaded.text_layer.split(PAGE_SEPARATOR)
    assert len(parts) == 3
    assert "First page" in parts[0]
    assert "Second page" in parts[1]
    assert "Third page" in parts[2]


def test_pdf_max_pages_renders_only_first_pages_but_reports_true_count(tmp_path):
    path = make_pdf(tmp_path / "long.pdf", ["Page one", "Page two", "Page three", "Page four", "Page five"])

    loaded = load_document(path, max_pages=2)

    assert loaded.page_count == 5
    assert len(loaded.images) == 2
    assert "Page one" in loaded.text_layer
    assert "Page two" in loaded.text_layer
    assert "Page five" not in loaded.text_layer


def test_pdf_without_text_layer_gives_none(tmp_path):
    path = make_pdf(tmp_path / "scan.pdf", [None])

    loaded = load_document(path)

    assert loaded.text_layer is None
    assert len(loaded.images) == 1


def test_pdf_whitespace_only_text_gives_none(tmp_path):
    path = make_pdf(tmp_path / "blank.pdf", ["   "])
    assert load_document(path).text_layer is None


def test_pdf_render_is_downscaled_to_max_side(tmp_path):
    # A US Letter page at 200 DPI is 1700 x 2200 px, so it must be scaled down to 1600 long side.
    path = make_pdf(tmp_path / "letter.pdf", ["Resize me"])

    loaded = load_document(path, dpi=200, max_side=1600)
    width, height = loaded.images[0].size

    assert max(width, height) == 1600
    assert height > width
    assert abs(width / height - 612 / 792) < 0.01


def test_pdf_small_render_is_not_upscaled(tmp_path):
    path = make_pdf(tmp_path / "small.pdf", ["Tiny"], size=(144, 72))

    loaded = load_document(path, dpi=72, max_side=1600)

    assert loaded.images[0].size == (144, 72)


def test_pdf_dpi_controls_render_size(tmp_path):
    path = make_pdf(tmp_path / "dpi.pdf", ["Dpi"], size=(72, 72))

    low = load_document(path, dpi=72)
    high = load_document(path, dpi=144)

    assert low.images[0].size == (72, 72)
    assert high.images[0].size == (144, 144)


def test_encrypted_pdf_raises_ingest_error(tmp_path):
    source = make_pdf(tmp_path / "plain.pdf", ["Secret"])
    encrypted = tmp_path / "locked.pdf"
    doc = pymupdf.open(source)
    doc.save(
        encrypted,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="user-secret",
        owner_pw="owner-secret",
    )
    doc.close()

    with pytest.raises(IngestError, match="password"):
        load_document(encrypted)


def test_corrupt_pdf_raises_ingest_error(tmp_path):
    path = write_bytes(tmp_path / "broken.pdf", b"%PDF-1.4\n%garbage that is not a pdf at all")
    with pytest.raises(IngestError):
        load_document(path)


# --- Raster images ---------------------------------------------------------------------


def test_png_is_loaded_as_rgb_without_resize_when_small(tmp_path):
    path = make_image(tmp_path / "06_discharge_summary_th.png", (120, 80), fmt="PNG")

    loaded = load_document(path)

    assert loaded.mime == "image/png"
    assert loaded.doc_id == "06_discharge_summary_th"
    assert loaded.page_count == 1
    assert loaded.text_layer is None
    assert loaded.images[0].mode == "RGB"
    assert loaded.images[0].size == (120, 80)


def test_large_image_is_downscaled_keeping_aspect_ratio(tmp_path):
    path = make_image(tmp_path / "wide.png", (3000, 1000), fmt="PNG")

    loaded = load_document(path, max_side=1600)

    assert loaded.images[0].size == (1600, 533)


def test_image_resize_respects_custom_max_side(tmp_path):
    path = make_image(tmp_path / "tall.jpg", (400, 1000), fmt="JPEG")
    loaded = load_document(path, max_side=500)
    assert loaded.images[0].size == (200, 500)


def test_rgba_is_composited_on_white(tmp_path):
    path = tmp_path / "alpha.png"
    image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))  # fully transparent
    image.paste((255, 0, 0, 255), (0, 0, 10, 10))  # opaque red on the left half
    image.save(path)

    loaded = load_document(path)
    rgb = loaded.images[0]

    assert rgb.mode == "RGB"
    assert rgb.getpixel((2, 5)) == (255, 0, 0)
    assert rgb.getpixel((15, 5)) == (255, 255, 255)


def test_grayscale_and_palette_images_become_rgb(tmp_path):
    gray = make_image(tmp_path / "gray.png", (16, 16), mode="L", color=128, fmt="PNG")
    palette = tmp_path / "palette.png"
    Image.new("P", (16, 16), 3).save(palette)

    assert load_document(gray).images[0].mode == "RGB"
    assert load_document(palette).images[0].mode == "RGB"


def test_palette_with_transparency_is_composited(tmp_path):
    path = tmp_path / "pal_alpha.png"
    image = Image.new("P", (8, 8), 0)
    image.info["transparency"] = 0
    image.save(path, transparency=0)

    rgb = load_document(path).images[0]

    assert rgb.mode == "RGB"
    assert rgb.getpixel((4, 4)) == (255, 255, 255)


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    Image.new("RGB", (40, 20), (10, 10, 10)).save(path, format="JPEG", exif=exif.tobytes())

    loaded = load_document(path)

    assert loaded.images[0].size == (20, 40)


def test_multi_frame_tiff_returns_pages_up_to_max_pages(tmp_path):
    path = tmp_path / "scan.tiff"
    frames = [Image.new("RGB", (60, 80), (i * 40, 0, 0)) for i in range(4)]
    frames[0].save(path, save_all=True, append_images=frames[1:])

    loaded = load_document(path, max_pages=3)

    assert loaded.mime == "image/tiff"
    assert loaded.page_count == 4
    assert len(loaded.images) == 3
    assert all(img.mode == "RGB" for img in loaded.images)


# --- Errors ----------------------------------------------------------------------------


def test_missing_file_raises_ingest_error(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        load_document(tmp_path / "does_not_exist.pdf")


def test_directory_raises_ingest_error(tmp_path):
    with pytest.raises(IngestError, match="Not a regular file"):
        load_document(tmp_path)


def test_empty_file_raises_ingest_error(tmp_path):
    with pytest.raises(IngestError, match="empty"):
        load_document(write_bytes(tmp_path / "empty.pdf", b""))


def test_plain_text_file_raises_unsupported(tmp_path):
    path = write_bytes(tmp_path / "notes.png", b"just some text, not an image")
    with pytest.raises(IngestError, match="Unsupported"):
        load_document(path)


def test_truncated_png_raises_ingest_error(tmp_path):
    path = write_bytes(tmp_path / "truncated.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    with pytest.raises(IngestError):
        load_document(path)


@pytest.mark.parametrize("kwargs", [{"max_side": 0}, {"dpi": 0}, {"max_pages": 0}])
def test_non_positive_options_are_rejected(tmp_path, kwargs):
    path = make_image(tmp_path / "ok.png", (10, 10), fmt="PNG")
    with pytest.raises(ValueError):
        load_document(path, **kwargs)


# --- Async wrapper ---------------------------------------------------------------------


async def test_aload_document_matches_sync_load(tmp_path):
    path = make_pdf(tmp_path / "async.pdf", ["Async text"])

    sync_result = load_document(path)
    async_result = await aload_document(path)

    assert async_result.doc_id == sync_result.doc_id
    assert async_result.mime == sync_result.mime
    assert async_result.text_layer == sync_result.text_layer
    assert async_result.images[0].size == sync_result.images[0].size


async def test_aload_document_propagates_errors(tmp_path):
    with pytest.raises(IngestError):
        await aload_document(tmp_path / "missing.png")


# --- Real fixtures --------------------------------------------------------------------


def _real_documents() -> list[Path]:
    if not DOCUMENT_TESTS_DIR.is_dir():
        return []
    return sorted(p for p in DOCUMENT_TESTS_DIR.iterdir() if p.is_file() and not p.name.startswith("."))


@pytest.mark.skipif(not DOCUMENT_TESTS_DIR.is_dir(), reason="document-tests/ folder is missing")
@pytest.mark.parametrize("path", _real_documents(), ids=lambda p: p.name)
def test_real_document_loads_within_limits(path):
    max_side = 1600
    loaded = load_document(path, max_side=max_side)

    assert loaded.doc_id == path.stem
    assert loaded.mime in {"application/pdf", "image/png", "image/jpeg", "image/webp", "image/tiff"}
    assert len(loaded.images) >= 1
    assert loaded.page_count >= len(loaded.images)
    for image in loaded.images:
        assert image.mode == "RGB"
        assert max(image.size) <= max_side
