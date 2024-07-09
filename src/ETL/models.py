from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ClassifyChunk(BaseModel):
    text: Optional[str] = None
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    creation_date: Optional[str] = None
    last_modified_date: Optional[str] = None
    fecha_publicacion_boe: Optional[str] = None
    orden: Optional[list[str]] = None
    real_decreto: Optional[list[str]] = None
    ministerios: Optional[list[str]] = None
    pdf_id: Optional[str] = None
    chunk_id: Optional[str] = None
    num_tokens: Optional[float] = None
    num_caracteres: Optional[float] = None
    label_1_label: Optional[str] = None
    label_1_score: Optional[float] = None
    label_2_label: Optional[str] = None
    label_2_score: Optional[float] = None
    label_3_label: Optional[str] = None
    label_3_score: Optional[float] = None
