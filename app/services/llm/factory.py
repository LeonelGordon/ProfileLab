from .base import BaseLLMClient
from .groq_client import GroqClient
from .openai_client import OpenAIClient


class LLMFactory:
    @staticmethod
    def create(provider: str) -> BaseLLMClient:
        normalized_provider = provider.strip().lower()

        if normalized_provider == "groq":
            return GroqClient()

        if normalized_provider == "openai":
            return OpenAIClient()

        raise ValueError(
            f"Provider no soportado: '{provider}'. Usá 'groq' o 'openai'."
        )