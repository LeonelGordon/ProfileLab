from typing import List, Optional, TypedDict

from app.schemas import LinkedInOutput, ProfileAnalysis


class ProfileLabState(TypedDict, total=False):
    # User input
    cv_file_path: str
    cv_text: str
    target_role: str
    provider: str

    # Agent 1 output
    profile_analysis: ProfileAnalysis

    # RAG output
    retrieved_chunks: List[str]

    # Agent 2 output
    linkedin_output: LinkedInOutput

    # Observability
    logs: List[str]

    # Error handling
    error: Optional[str]