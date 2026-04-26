from app.agents import LinkedInStrategistAgent


def generate_linkedin_profile_node(state):
    try:
        agent = LinkedInStrategistAgent(provider=state["provider"])

        linkedin_output = agent.run(
            profile_analysis=state["profile_analysis"],
            retrieved_chunks=state["retrieved_chunks"],
            target_role=state["target_role"],
        )

        logs = state.get("logs", [])
        logs.append("[generate_linkedin_profile] OK: optimización de LinkedIn generada.")
        llm_trace = getattr(agent.llm, "last_structured_trace", None)
        if llm_trace:
            logs.extend(llm_trace)

        return {
            "linkedin_output": linkedin_output,
            "logs": logs,
        }

    except Exception as e:
        logs = state.get("logs", [])
        logs.append(
            f"[generate_linkedin_profile] ERROR: falló la generación de la optimización ({str(e)})."
        )
        llm_trace = getattr(agent.llm, "last_structured_trace", None)
        if llm_trace:
            logs.extend(llm_trace)

        return {
            "error": str(e),
            "logs": logs,
        }