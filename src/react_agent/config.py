"""Configuración de la aplicación cargada desde variables de entorno."""

from pathlib import Path

from pydantic import BaseModel, Field, SecretStr

PLACEHOLDER_API_KEY = "replace_with_your_openai_api_key"


class ConfigurationError(RuntimeError):
    """Falta o es inválida una variable de entorno necesaria."""


class Settings(BaseModel):
    """Valida toda la configuración que necesita el agente."""

    openai_api_key: SecretStr
    openai_model: str = "gpt-4.1-mini"
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")
    trace_output_path: Path = Path("traces/latest_trace.json")
    thread_id: str = Field(default="cliente-102-demo", min_length=1)
    recursion_limit: int = Field(default=10, ge=2, le=50)

    @classmethod
    def from_environment(cls) -> "Settings":
        """Construye la configuración sin exponer la API key en logs."""

        import os

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or api_key == PLACEHOLDER_API_KEY:
            raise ConfigurationError(
                "Falta OPENAI_API_KEY. Copiá .env.example como .env y agregá tu clave."
            )

        return cls(
            openai_api_key=SecretStr(api_key),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            checkpoint_db_path=Path(
                os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
            ),
            trace_output_path=Path(
                os.getenv("TRACE_OUTPUT_PATH", "traces/latest_trace.json")
            ),
            thread_id=os.getenv("THREAD_ID", "cliente-102-demo"),
        )
