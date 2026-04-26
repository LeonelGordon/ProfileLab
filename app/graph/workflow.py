from langgraph.graph import StateGraph, START, END

from app.graph import ProfileLabState
from app.nodes import (
    analyze_profile_node,
    generate_linkedin_profile_node,
    parse_cv_node,
    retrieve_best_practices_node,
)


def build_graph():
    graph = StateGraph(ProfileLabState)

    graph.add_node("parse_cv", parse_cv_node)
    graph.add_node("analyze_profile", analyze_profile_node)
    graph.add_node("retrieve_best_practices", retrieve_best_practices_node)
    graph.add_node("generate_linkedin_profile", generate_linkedin_profile_node)

    graph.add_edge(START, "parse_cv")
    graph.add_edge("parse_cv", "analyze_profile")
    graph.add_edge("analyze_profile", "retrieve_best_practices")
    graph.add_edge("retrieve_best_practices", "generate_linkedin_profile")
    graph.add_edge("generate_linkedin_profile", END)

    return graph.compile()