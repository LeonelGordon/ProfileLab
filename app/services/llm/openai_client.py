import os
from typing import Type, TypeVar

from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from .base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("No se encontró la API Key de OpenAI (OPENAI_API_KEY)")

        self.model = model

        try:
            self.client = ChatOpenAI(
                api_key=self.api_key,
                model=self.model,
                temperature=0.2,
            )
        except Exception as e:
            raise RuntimeError(f"Error al inicializar el cliente de OpenAI: {e}") from e

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.invoke(prompt)

            if not response.content:
                raise ValueError("El modelo devolvió una respuesta vacía")

            return str(response.content).strip()

        except Exception as e:
            raise RuntimeError(f"Error al generar texto con OpenAI: {e}") from e

    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        try:
            structured_llm = self.client.with_structured_output(schema)
            response = structured_llm.invoke(prompt)

            if response is None:
                raise ValueError("El modelo no devolvió una respuesta estructurada válida")

            return response

        except Exception as e:
            raise RuntimeError(f"Error al generar salida estructurada con OpenAI: {e}") from e