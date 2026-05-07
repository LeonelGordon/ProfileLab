from app.rag import get_persistent_chroma_rag


def retrieve_best_practices_node(state):
    try:
        profile_analysis = state["profile_analysis"]
        target_role = state["target_role"]

        query = f"""
        LinkedIn profile optimization for role: {target_role}.
        Candidate focus: {profile_analysis.current_focus}.
        Seniority: {profile_analysis.seniority}.
        Relevant keywords: {", ".join(profile_analysis.linkedin_keywords)}.
        Need decision patterns for:
        - headline optimization
        - about section
        - skills selection
        - positioning strategy
        - content strategy
        """

        rag = get_persistent_chroma_rag()
        retrieved_chunks = rag.retrieve(query=query, top_k=3)

        logs = state.get("logs", [])
        logs.append("[retrieve_best_practices] OK: recuperación RAG completada.")

        return {
            "retrieved_chunks": retrieved_chunks,
            "logs": logs,
        }

    except Exception as e:
        logs = state.get("logs", [])
        logs.append(f"[retrieve_best_practices] ERROR: falló la recuperación RAG ({str(e)}).")

        return {
            "error": str(e),
            "logs": logs,
        }