from typing import Any, List, Optional, TypedDict

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

    # Iteration / refinement (refine_only solo en invoke acortado vía Streamlit / API)
    refine_only: Optional[bool]
    user_feedback: Optional[str]
    refined_output: Optional[LinkedInOutput]
    iteration_history: List[dict[str, Any]]

    # Observability
    logs: List[str]

    # Error handling
    error: Optional[str]