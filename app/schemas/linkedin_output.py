from typing import List
from pydantic import BaseModel


class LinkedInOutput(BaseModel):
    headline: str
    about: str
    suggested_skills: List[str]
    seo_recommendations: List[str]
    content_recommendations: List[str]