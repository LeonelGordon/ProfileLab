from typing import List
from pydantic import BaseModel


class ProfileAnalysis(BaseModel):
    candidate_summary: str
    seniority: str
    current_focus: str
    core_skills: List[str]
    strengths: List[str]
    linkedin_keywords: List[str]