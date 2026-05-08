from app.graph.workflow import build_graph
from app.config import bootstrap

bootstrap()

def main():
    graph = build_graph()

    initial_state = {
        "cv_file_path": "docs/LeonelGordon.pdf",
        "target_role": "Frontend Developer",
        "provider": "groq",
        "logs": [],
    }

    result = graph.invoke(initial_state)

    if result.get("error"):
        print("\n=== ERROR ===")
        print(result["error"])

    print("\n=== LINKEDIN OUTPUT ===")
    linkedin_output = result.get("linkedin_output")

    if linkedin_output:
        print(linkedin_output.model_dump_json(indent=2))
    else:
        print("No LinkedIn output generated.")

    print("\n=== LOGS ===")
    for log in result.get("logs", []):
        print(f"- {log}")


if __name__ == "__main__":
    main()