import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def unzip_pdf(zip_file_path: Path) -> Path | None:
    """Extract PDF from zip file and return path to temporary PDF file."""
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        pdf_files = [name for name in zip_ref.namelist() if name.endswith(".pdf")]
        if not pdf_files:
            return None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(zip_ref.read(pdf_files[0]))
            return Path(temp_file.name)


def extract_text_from_pdf(pdf_data: bytes) -> str | None:
    """Extract text content from PDF bytes."""
    try:
        pdf_reader = PdfReader(BytesIO(pdf_data))
        text_pages = [page.extract_text() or "" for page in pdf_reader.pages]
        text = "".join(text_pages).strip()
        return text if text else None
    except Exception:
        return None


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        pdf_reader = PdfReader(str(pdf_path))
        return len(pdf_reader.pages)
    except Exception:
        return 0
