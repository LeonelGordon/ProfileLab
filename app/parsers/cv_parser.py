from pathlib import Path

from pypdf import PdfReader
from docx import Document


class CVParserError(Exception):
    pass


def extract_cv_text(file_path: str) -> str:
    """
    Extrae texto de un CV.

    Soporta:
    - PDF (.pdf)
    - Word (.docx)
    """
    path = Path(file_path)

    if not path.exists():
        raise CVParserError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        text = _extract_text_from_pdf(path)
    elif extension == ".docx":
        text = _extract_text_from_docx(path)
    else:
        raise CVParserError("Only PDF and DOCX are supported.")

    text = _clean_text(text)

    if not text.strip():
        if extension == ".pdf":
            raise CVParserError(
                "No se pudo extraer texto del PDF. "
                "Este archivo probablemente está escaneado (imagen) o no tiene capa de texto. "
                "Subí un PDF 'nativo' (exportado desde Word/Google Docs) o convertí el PDF a "
                "'Searchable PDF' con OCR y volvé a intentar."
            )
        raise CVParserError("Empty content.")

    return text


def _extract_text_from_pdf(path: Path) -> str:
    """Extrae texto desde PDF"""
    try:
        reader = PdfReader(str(path))
        texts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)

        return "\n".join(texts)

    except Exception as e:
        raise CVParserError(f"PDF error: {str(e)}")


def _extract_text_from_docx(path: Path) -> str:
    """Extrae texto desde DOCX"""
    try:
        doc = Document(str(path))
        texts = [p.text for p in doc.paragraphs if p.text]

        return "\n".join(texts)

    except Exception as e:
        raise CVParserError(f"DOCX error: {str(e)}")


def _clean_text(text: str) -> str:
    """Limpieza básica de texto"""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)