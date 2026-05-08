from __future__ import annotations

import os
from typing import Any, Mapping


def _try_load_dotenv() -> None:
    """
    Carga variables desde `.env` en local, si `python-dotenv` está disponible.
    No pisa variables ya definidas (override=False) para no romper despliegues.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    load_dotenv(override=False)


def _try_get_streamlit_secrets() -> Mapping[str, Any]:
    """
    Lee `st.secrets` si Streamlit está disponible y el contexto lo permite.
    En local (sin secrets.toml) devuelve dict vacío.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return {}

    try:
        # En Streamlit, acceder/iterar `st.secrets` puede disparar el parseo y,
        # si no existe `secrets.toml`, lanza StreamlitSecretNotFoundError.
        secrets = st.secrets
        if hasattr(secrets, "to_dict"):
            return secrets.to_dict()  # type: ignore[no-any-return]
        return dict(secrets)  # type: ignore[arg-type]
    except Exception:
        return {}


def bootstrap() -> None:
    """
    Inicializa configuración de forma compatible:
    - Local: carga `.env` si existe.
    - Producción (Streamlit Cloud): permite consumir `st.secrets`.

    Para mantener compatibilidad con el código actual, también “inyecta”
    secretos de Streamlit en `os.environ` si la variable aún no existe.
    """
    _try_load_dotenv()

    secrets = _try_get_streamlit_secrets()
    if not secrets:
        return

    # Claves esperadas por el código actual
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        if key in secrets and secrets.get(key):
            os.environ.setdefault(key, str(secrets.get(key)))


def get_secret(name: str, *, default: str | None = None, required: bool = False) -> str | None:
    """
    Obtiene un secreto por orden de prioridad:
    1) variable de entorno
    2) Streamlit Secrets
    3) default
    """
    value = os.getenv(name)
    if value:
        return value

    secrets = _try_get_streamlit_secrets()
    if name in secrets and secrets.get(name):
        return str(secrets.get(name))

    if default is not None:
        return default

    if required:
        raise ValueError(f"Falta el secreto requerido: {name}")

    return None

