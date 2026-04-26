from pathlib import Path
from typing import List


def load_markdown_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor a 0")

    if overlap < 0:
        raise ValueError("overlap no puede ser negativo")

    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    chunks = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_and_chunk_markdown(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[str]:
    text = load_markdown_file(file_path)
    return chunk_text(text=text, chunk_size=chunk_size, overlap=overlap)