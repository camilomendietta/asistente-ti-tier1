import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Parámetros del modelo de lenguaje local y control de generación."""
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_new_tokens: int = 180          # Margen suficiente para cerrar oraciones sin relleno extenso
    temperature: float = 0.0           # Determinista (0.0) para priorizar fidelidad al manual
    do_sample: bool = False            # Desactiva muestreo aleatorio en modo determinista
    repetition_penalty: float = 1.2    # Reduce repeticiones y bucles en respuestas cortas
    max_history_messages: int = 6      # Ventana móvil de memoria a corto plazo (3 turnos)


@dataclass(frozen=True)
class RagConfig:
    """Parámetros de ingesta documental, fragmentación y recuperación vectorial."""
    pdf_path: str = "manual_soporte_tier1.pdf"
    persist_directory: str = os.path.abspath("./chroma_db_persistente")
    collection_name: str = "it_support_kb_pdf"
    chunk_size: int = 350              # Tamaño de fragmento adaptado a contexto compacto
    chunk_overlap: int = 50            # Solapamiento para preservar continuidad entre cortes
    top_k: int = 1                     # Recupera el fragmento más relevante para evitar mezclas


@dataclass(frozen=True)
class GovernanceConfig:
    """Rutas locales para perfilado anónimo y auditoría sin PII."""
    profile_filepath: str = "perfil_usuario.json"
    log_filepath: str = "registro_soporte_ti.jsonl"
    default_os: str = "Windows 11"
    default_format: str = "Pasos numerados claros"
    min_valid_response_len: int = 15   # Umbral mínimo para marcar interacción como EXITO


@dataclass(frozen=True)
class AppConfig:
    """Contenedor maestro de configuración del asistente TI."""
    app_title: str = "Asistente de Soporte TI - Nivel 1"
    app_subtitle: str = "Consulta guiada basada en documentación oficial corporativa (sin envío de datos a APIs externas)"
    model: ModelConfig = ModelConfig()
    rag: RagConfig = RagConfig()
    governance: GovernanceConfig = GovernanceConfig()


# Instancia global lista para importar en otros módulos (backend, app.py, etc.)
APP_CONFIG = AppConfig()