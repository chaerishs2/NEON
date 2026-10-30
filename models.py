from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class Status(str, Enum):
    processing = "processing"
    answered_by_ai = "answered_by_ai"
    pending_nurse = "pending_nurse"
    responded = "responded"
    handled_directly = "handled_directly"
    emergency = "emergency"
    emergency_acknowledged = "emergency_acknowledged"


class Category(str, Enum):
    simple_info = "simple_info"
    needs_nurse = "needs_nurse"
    emergency = "emergency"


class RequestIn(BaseModel):
    patient_id: str
    text: str
    input_type: Literal["text", "voice"] = "text"


class Visit(BaseModel):
    needed: bool = False
    eta_min: Optional[int] = None


class RespondIn(BaseModel):
    action: Literal["approve", "edit", "direct"]
    text: str = ""
    visit: Visit = Visit()
    nurse_name: str


class AckIn(BaseModel):
    nurse_name: str = "이지은"
