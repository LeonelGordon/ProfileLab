from app.parsers import extract_cv_text


def parse_cv_node(state):
    try:
        cv_file_path = state["cv_file_path"]

        cv_text = extract_cv_text(cv_file_path)

        logs = state.get("logs", [])
        logs.append("[parse_cv] OK: CV parseado y texto extraído.")

        return {
            "cv_text": cv_text,
            "logs": logs,
        }

    except Exception as e:
        logs = state.get("logs", [])
        logs.append(f"[parse_cv] ERROR: no se pudo parsear el CV ({str(e)}).")

        return {
            "error": str(e),
            "logs": logs,
        }