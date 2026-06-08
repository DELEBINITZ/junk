from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = None


class Citation(BaseModel):
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    source: str = ""


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    trace_id: str = ""
    citations: list[Citation] = []
    agents_used: list[str] = []
    blocked: bool = False
    block_reason: str = ""
    is_complex: bool = False


class SessionInfo(BaseModel):
    session_id: str
    title: str = ""
    message_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
