from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    file_name: str
    date: str
    transcription_text: str
    model_name: str
    provider_name: str
    transcription_time: float = 0.0
    error: Optional[str] = None
    conversion_error: Optional[str] = None
