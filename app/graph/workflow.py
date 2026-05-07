from langgraph.graph import StateGraph, START, END

from app.graph import ProfileLabState
from app.nodes import (
    analyze_profile_node,
    generate_linkedin_profile_node,
    output_refiner_node,
    parse_cv_node,
    retrieve_best_practices_node,
)


def route_entry(state: ProfileLabState) -> str:
    if state.get("refine_only"):
        return "refine_only"
    return "full"


def should_refine_output(state: ProfileLabState) -> str:
    feedback = (state.get("user_feedback") or "").strip()
    return "refine" if feedback else "end"


def build_graph():
    graph = StateGraph(ProfileLabState)

    graph.add_node("parse_cv", parse_cv_node)
    graph.add_node("analyze_profile", analyze_profile_node)
    graph.add_node("retrieve_best_practices", retrieve_best_practices_node)
    graph.add_node("generate_linkedin_profile", generate_linkedin_profile_node)
    graph.add_node("output_refiner", output_refiner_node)

    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "full": "parse_cv",
            "refine_only": "output_refiner",
        },
    )
    graph.add_edge("parse_cv", "analyze_profile")
    graph.add_edge("analyze_profile", "retrieve_best_practices")
    graph.add_edge("retrieve_best_practices", "generate_linkedin_profile")

    graph.add_conditional_edges(
        "generate_linkedin_profile",
        should_refine_output,
        {
            "refine": "output_refiner",
            "end": END,
        },
    )

    graph.add_edge("output_refiner", END)

    return graph.compile()