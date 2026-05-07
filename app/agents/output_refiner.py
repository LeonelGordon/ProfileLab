from app.schemas import LinkedInOutput, ProfileAnalysis
from app.services.llm import LLMFactory


class OutputRefinerAgent:
    """Ajusta el LinkedInOutput previo según feedback del usuario, mismo esquema Pydantic."""

    def __init__(self, provider: str) -> None:
        self.llm = LLMFactory.create(provider)

    def run(
        self,
        *,
        linkedin_output: LinkedInOutput,
        user_feedback: str,
        target_role: str,
        profile_analysis: ProfileAnalysis,
    ) -> LinkedInOutput:
        current = linkedin_output.model_dump_json(indent=2)
        profile_json = profile_analysis.model_dump_json(indent=2)

        prompt = f"""
Sos el mismo especialista en LinkedIn SEO y personal branding que generó la propuesta inicial.

Recibís:
1. El output de LinkedIn ya generado (headline, about, skills, recomendaciones).
2. Feedback concreto del usuario sobre qué cambiar o mejorar.
3. El rol objetivo y el análisis del perfil (fuente de verdad; no inventar hechos).

---

ROL OBJETIVO
{target_role}

---

ANÁLISIS DEL PERFIL (no contradecir; no agregar experiencia inventada)
{profile_json}

---

OUTPUT ACTUAL (a refinar)
{current}

---

FEEDBACK DEL USUARIO
{user_feedback}

---

TAREA

- Aplicá el feedback de forma explícita donde tenga sentido.
- Mantené coherencia con el rol objetivo y el análisis del perfil.
- No inventes empleos, métricas, certificaciones ni tecnologías que no estén respaldados por el análisis.
- Conservá el mismo nivel de especificidad; evitá volver al contenido genérico.
- Si el feedback pide algo imposible sin inventar datos, ajustá lo máximo posible sin falsear el perfil y mantené el resto alineado al análisis.

Devolvé el resultado completo con el mismo esquema estructurado (todos los campos requeridos).
"""

        return self.llm.generate_structured(prompt, LinkedInOutput)


def output_refiner_node(state):
    """Nodo LangGraph: refina sobre `refined_output` previo si existe, si no sobre `linkedin_output`."""
    try:
        feedback = (state.get("user_feedback") or "").strip()
        if not feedback:
            logs = state.get("logs", [])
            logs.append("[output_refiner] SKIP: user_feedback vacío; no se refinó.")
            return {"logs": logs, "refine_only": False}

        base_output = state.get("refined_output") or state.get("linkedin_output")
        if base_output is None:
            raise ValueError("Falta linkedin_output (y no hay refined_output previo) en el estado.")

        refine_source = "refined_output" if state.get("refined_output") else "linkedin_output"

        agent = OutputRefinerAgent(provider=state["provider"])
        refined = agent.run(
            linkedin_output=base_output,
            user_feedback=feedback,
            target_role=state["target_role"],
            profile_analysis=state["profile_analysis"],
        )

        logs = state.get("logs", [])
        logs.append("[output_refiner] OK: salida refinada según feedback del usuario.")
        llm_trace = getattr(agent.llm, "last_structured_trace", None)
        if llm_trace:
            logs.extend(llm_trace)

        history = list(state.get("iteration_history") or [])
        history.append(
            {
                "user_feedback": feedback,
                "source": refine_source,
                "refined_output": refined.model_dump(),
            }
        )

        return {
            "refined_output": refined,
            "logs": logs,
            "iteration_history": history,
            "refine_only": False,
        }

    except Exception as e:
        logs = state.get("logs", [])
        logs.append(f"[output_refiner] ERROR: falló el refinamiento ({str(e)}).")
        agent = locals().get("agent")
        if agent is not None:
            llm_trace = getattr(agent.llm, "last_structured_trace", None)
            if llm_trace:
                logs.extend(llm_trace)

        return {
            "error": str(e),
            "logs": logs,
            "refine_only": False,
        }
