from app.agents import ProfileAnalystAgent


def analyze_profile_node(state):
    try:
        agent = ProfileAnalystAgent(provider=state["provider"])

        profile_analysis = agent.run(
            cv_text=state["cv_text"],
            target_role=state["target_role"],
        )

        logs = state.get("logs", [])
        logs.append("[analyze_profile] OK: análisis de perfil completado.")
        llm_trace = getattr(agent.llm, "last_structured_trace", None)
        if llm_trace:
            logs.extend(llm_trace)

        return {
            "profile_analysis": profile_analysis,
            "logs": logs,
        }

    except Exception as e:
        logs = state.get("logs", [])
        logs.append(f"[analyze_profile] ERROR: falló el análisis de perfil ({str(e)}).")
        llm_trace = getattr(agent.llm, "last_structured_trace", None)
        if llm_trace:
            logs.extend(llm_trace)

        return {
            "error": str(e),
            "logs": logs,
        }