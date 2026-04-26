import os
import re
from typing import Type, TypeVar

from pydantic import BaseModel
from langchain_groq import ChatGroq

from .base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)


class GroqClient(BaseLLMClient):
    def __init__(self, model: str = "openai/gpt-oss-20b") -> None:
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("No se encontró la API Key de Groq (GROQ_API_KEY)")

        self.model = model
        self.last_structured_trace: list[str] = []

        try:
            self.client = ChatGroq(
                api_key=self.api_key,
                model=self.model,
                temperature=0.2,
            )
        except Exception as e:
            raise RuntimeError(f"Error al inicializar el cliente de Groq: {e}") from e

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.invoke(prompt)

            if not response.content:
                raise ValueError("El modelo devolvió una respuesta vacía")

            return str(response.content).strip()

        except Exception as e:
            raise RuntimeError(f"Error al generar texto con Groq: {e}") from e

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        # Vi que el modelo en Groq a veces no responde bien con salida estructurada (tool calling),
        # por eso agregamos reintentos y un fallback de modelo para maximizar la chance de obtenerla.
        self.last_structured_trace = []
        # Preferimos gpt-oss-20b por consistencia con structured output.
        # Mantenemos fallbacks para cubrir inestabilidad / rate limits.
        fallback_models = [
            self.model,
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
        ]

        def _sanitize_error_message(exc: Exception, max_len: int = 220) -> str:
            """
            Recorta y sanitiza errores para trazabilidad (evita exponer IDs / payloads largos).
            """
            msg = str(exc)
            msg = re.sub(r"org_[A-Za-z0-9]+", "org_[REDACTED]", msg)
            msg = re.sub(r"sk-[A-Za-z0-9]+", "sk-[REDACTED]", msg)
            msg = re.sub(r"https?://\\S+", "[URL_REDACTED]", msg)
            msg = re.sub(r"\\s+", " ", msg).strip()
            if len(msg) > max_len:
                msg = msg[: max_len - 1] + "…"
            return msg

        def _is_retryable_structured_error(exc: Exception) -> bool:
            msg = str(exc).lower()
            return any(
                token in msg
                for token in (
                    "tool_use_failed",
                    "failed to call a function",
                    "invalid_request_error",
                    "function_call",
                    "tool calling",
                    "rate_limit",
                    "rate limit",
                    "429",
                )
            )

        last_exc: Exception | None = None

        for model in fallback_models:
            # Para salida estructurada preferimos temperatura 0 (más determinista y “obediente”).
            client = ChatGroq(
                api_key=self.api_key,
                model=model,
                temperature=0.0,
            )
            structured_llm = client.with_structured_output(schema)

            for _attempt in range(2):
                try:
                    self.last_structured_trace.append(
                        f"[llm][structured] intento={_attempt + 1} modelo={model}"
                    )
                    response = structured_llm.invoke(prompt)

                    if response is None:
                        raise ValueError(
                            "El modelo no devolvió una respuesta estructurada válida"
                        )

                    self.last_structured_trace.append(
                        f"[llm][structured] OK modelo={model}"
                    )
                    return response

                except Exception as e:
                    last_exc = e
                    self.last_structured_trace.append(
                        f"[llm][structured] fallo modelo={model} ({_sanitize_error_message(e)})"
                    )
                    if not _is_retryable_structured_error(e):
                        break

        raise RuntimeError(
            f"Error al generar salida estructurada con Groq: {last_exc}"
        ) from last_exc