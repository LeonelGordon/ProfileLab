import os
import tempfile
import html
import re

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from app.graph.workflow import build_graph

load_dotenv()


PROGRESS_STEPS = [
    ("parse_cv", "01", "Parser", "Leyendo y limpiando el CV."),
    ("analyze_profile", "02", "Profile Analyst", "Analizando seniority, foco y keywords."),
    ("retrieve_best_practices", "03", "RAG Retriever", "Recuperando contexto externo."),
    ("generate_linkedin_profile", "04", "Strategist", "Generando optimización final."),
]


def render_progress(placeholder, completed, active, errors=None):
    errors = errors or {}
    first_error_seen = False
    rows = []

    for key, num, title, desc in PROGRESS_STEPS:
        err_msg = errors.get(key)
        error_html = ""

        if err_msg and not first_error_seen:
            state = "error"
            status_label = "Error"
            first_error_seen = True
            error_html = f'<div class="progress-error">{safe_text(err_msg)}</div>'
        elif first_error_seen:
            state = "skipped"
            status_label = "No ejecutado"
        elif key in completed:
            state = "done"
            status_label = "Listo"
        elif key == active:
            state = "running"
            status_label = "Procesando"
        else:
            state = "pending"
            status_label = "En espera"

        rows.append(
            f'<div class="progress-step {state}">'
            f'<div class="progress-num">{num}</div>'
            f'<div class="progress-info">'
            f'<div class="progress-title">{title}</div>'
            f'<div class="progress-desc">{desc}</div>'
            f'{error_html}'
            f'</div>'
            f'<div class="progress-status">{status_label}</div>'
            f'</div>'
        )

    placeholder.markdown(
        '<div class="progress-wrap">' + "".join(rows) + '</div>',
        unsafe_allow_html=True,
    )


def run_profilelab_stream(cv_file_path: str, target_role: str, provider: str, on_node_done):
    graph = build_graph()

    initial_state = {
        "cv_file_path": cv_file_path,
        "target_role": target_role,
        "provider": provider,
        "logs": [],
    }

    accumulated = dict(initial_state)

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            node_error = None
            if isinstance(update, dict):
                accumulated.update(update)
                node_error = update.get("error")
            on_node_done(node_name, node_error)

    return accumulated


def safe_text(value):
    if value is None:
        return ""
    return html.escape(str(value))


def simplify_visible_error(value) -> str:
    """
    Simplifica errores para UI (sin exponer payloads largos, org ids o tokens).
    Regresa una única palabra clave: rate_limit | invalid_structure | unknown_error
    """
    msg = ("" if value is None else str(value)).lower()

    if any(token in msg for token in ("rate limit", "rate_limit", "429")):
        return "rate_limit"

    if any(
        token in msg
        for token in (
            # structured output / tool calling / parsing
            "structured output",
            "with_structured_output",
            "tool calling",
            "tool_use_failed",
            "failed to call a function",
            "function_call",
            "output parser",
            "jsondecodeerror",
            "pydantic",
            "schema",
        )
    ):
        return "invalid_structure"

    return "unknown_error"


def _render_visible_log_lines(logs: list[str]) -> list[str]:
    """
    Renderiza logs para UI con limpieza de errores del LLM.
    Mantiene logs internos intactos; solo transforma la visualización.
    """
    attempt_re = re.compile(r"^\[llm\]\[structured\]\s+intento=(\d+)\s+modelo=(.+)$")
    fail_re = re.compile(r"^\[llm\]\[structured\]\s+fallo\s+modelo=(.+?)(?:\s+\((.*)\))?$")
    ok_re = re.compile(r"^\[llm\]\[structured\]\s+OK\s+modelo=(.+)$")

    out: list[str] = []
    pending_attempt: dict | None = None
    last_model: str | None = None

    for raw in logs:
        line = "" if raw is None else str(raw)

        m_attempt = attempt_re.match(line)
        if m_attempt:
            model = m_attempt.group(2).strip()
            if last_model is not None and model != last_model:
                out.append(f"[llm][structured] fallback modelo={model}")
            last_model = model
            pending_attempt = {"model": model, "line": line}
            out.append(line)
            continue

        m_fail = fail_re.match(line)
        if m_fail:
            model = (m_fail.group(1) or "").strip()
            err_detail = m_fail.group(2) or ""
            keyword = simplify_visible_error(err_detail or line)

            if pending_attempt and pending_attempt.get("model") == model and out:
                out[-1] = f"{pending_attempt['line']} → fallo: {keyword}"
                pending_attempt = None
            else:
                prefix = f"[llm][structured] modelo={model}" if model else "[llm][structured]"
                out.append(f"{prefix} → fallo: {keyword}")
            continue

        m_ok = ok_re.match(line)
        if m_ok:
            pending_attempt = None
            model = m_ok.group(1).strip()
            last_model = model
            out.append(line)
            continue

        pending_attempt = None
        out.append(line)

    return out


def scroll_to_anchor(anchor_id: str, delay_ms: int = 120):
    components.html(
        f"""
        <script>
        (function() {{
            const scroll = () => {{
                const doc = window.parent && window.parent.document
                    ? window.parent.document
                    : document;
                const target = doc.getElementById('{anchor_id}');
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start',
                    }});
                }}
            }};
            setTimeout(scroll, {delay_ms});
        }})();
        </script>
        """,
        height=0,
    )


st.set_page_config(
    page_title="ProfileLab",
    page_icon="app/assets/icon.png",
    layout="wide",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F5F2EA;
        --surface: #FAF8F3;
        --ink: #0F172A;
        --muted: #475569;
        --soft-muted: #64748B;
        --line: rgba(15, 23, 42, 0.22);
        --dark: #0F172A;
        --input-bg: #FFFFFF;
        --success: #16A34A;
        --warning: #CA8A04;
        --danger: #DC2626;
        --danger-bg: rgba(220, 38, 38, 0.08);
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif !important;
    }

    header[data-testid="stHeader"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }

    .block-container {
        max-width: 1120px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 4rem !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
    }

    * {
        color-scheme: light !important;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stApp label,
    .block-container p,
    .block-container span {
        color: var(--ink);
    }

    .topbar {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        align-items: center;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink);
        margin-bottom: 130px;
    }

    .topbar div:nth-child(2) {
        text-align: center;
    }

    .topbar div:nth-child(3) {
        text-align: right;
    }

    .brand-small {
        font-weight: 800;
    }

    .hero-title {
        font-size: 104px;
        line-height: 0.86;
        font-weight: 800;
        letter-spacing: -0.085em;
        color: var(--ink);
        border-bottom: 1px solid var(--ink);
        max-width: 900px;
        margin-bottom: 18px;
    }

    .hero-subtitle {
        font-size: 34px;
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -0.05em;
        color: var(--ink);
        margin-bottom: 24px;
    }

    .hero-copy {
        font-size: 15px;
        max-width: 520px;
        color: var(--muted);
        line-height: 1.7;
        margin-bottom: 34px;
    }

    .hero-cta {
        display: inline-block;
        background: var(--dark);
        color: #FFFFFF !important;
        text-decoration: none !important;
        border-radius: 999px;
        padding: 14px 26px;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 120px;
    }

    .agent-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 24px;
        margin-bottom: 90px;
    }

    .agent-card {
        border-top: 1px solid var(--line);
        padding-top: 16px;
        min-height: 96px;
    }

    .agent-card-title {
        font-size: 20px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--ink);
        margin-bottom: 12px;
    }

    .agent-card-copy {
        font-size: 13px;
        color: var(--muted);
        line-height: 1.45;
    }

    .section {
        border-top: 1px solid var(--line);
        padding-top: 34px;
        margin-top: 20px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: var(--ink);
        margin-bottom: 12px;
    }

    .section-caption {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 24px;
    }

    .result-section {
        border-top: 1px solid var(--line);
        padding-top: 30px;
        margin-top: 42px;
    }

    .result-block {
        display: grid;
        grid-template-columns: 220px 1fr;
        gap: 28px;
        border-top: 1px solid var(--line);
        padding: 20px 0;
    }

    .result-label {
        font-size: 14px;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    .result-value {
        font-size: 15px;
        color: var(--muted);
        line-height: 1.65;
        white-space: pre-wrap;
    }

    .list-item {
        margin-bottom: 8px;
        color: var(--muted);
        line-height: 1.5;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 18px;
    }

    .metric-card {
        border-top: 1px solid var(--line);
        padding-top: 14px;
    }

    .metric-label {
        font-size: 12px;
        color: var(--soft-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 18px;
        font-weight: 800;
        color: var(--ink);
    }

    div.stButton > button,
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #FFFFFF !important;
    }

    div.stButton > button {
        background: var(--dark) !important;
        border-radius: 999px !important;
        padding: 0.8rem 1.6rem !important;
        border: 1px solid var(--dark) !important;
        font-weight: 700 !important;
        box-shadow: none !important;
        transition: background 0.15s ease, transform 0.15s ease;
    }

    div.stButton > button:hover,
    div.stButton > button:focus,
    div.stButton > button:active {
        background: #1E293B !important;
        border: 1px solid #1E293B !important;
        color: #FFFFFF !important;
        outline: none !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--input-bg) !important;
        border: 1px dashed rgba(15, 23, 42, 0.35) !important;
        border-radius: 18px !important;
    }

    [data-testid="stFileUploader"] *:not(svg):not(path):not(section) {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--ink) !important;
    }

    [data-testid="stFileUploader"] small {
        color: var(--muted) !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] svg,
    [data-testid="stFileUploader"] svg path {
        fill: var(--ink) !important;
        color: var(--ink) !important;
    }

    [data-testid="stFileUploader"] button {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--ink) !important;
        border: 1px solid rgba(15, 23, 42, 0.25) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] section > button {
        border: 1px solid rgba(15, 23, 42, 0.25) !important;
    }

    [data-testid="stFileUploader"] section ~ * button,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {
        border: none !important;
        opacity: 0.65;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label {
        color: var(--ink) !important;
        font-weight: 700 !important;
        font-size: 13px !important;
    }

    [data-testid="stTextInput"] div[data-baseweb="input"],
    [data-testid="stTextInput"] div[data-baseweb="base-input"] {
        background: var(--input-bg) !important;
        border: 1px solid rgba(15, 23, 42, 0.22) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }

    [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
    [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
        border-color: var(--ink) !important;
    }

    [data-testid="stTextInput"] input {
        background: transparent !important;
        color: var(--ink) !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 12px 14px !important;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: var(--input-bg) !important;
        color: var(--ink) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(15, 23, 42, 0.22) !important;
        box-shadow: none !important;
        min-height: 46px;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        fill: var(--ink) !important;
        color: var(--ink) !important;
    }

    div[role="listbox"] {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
    }

    div[role="option"] {
        color: var(--ink) !important;
    }

    div[data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }

    .metrics-grid + div[data-testid="stExpander"],
    .section-gap + div[data-testid="stExpander"] {
        margin-top: 34px !important;
    }

    .section-gap {
        height: 28px;
    }

    .scroll-anchor {
        position: relative;
        top: -24px;
        height: 0;
        pointer-events: none;
        scroll-margin-top: 24px;
    }

    div[data-testid="stExpander"] details {
        background: transparent !important;
        border: none !important;
    }

    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details[open] summary,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary[aria-expanded="true"],
    div[data-testid="stExpander"] summary[aria-expanded="false"] {
        background: transparent !important;
        color: var(--ink) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary div {
        color: var(--ink) !important;
        background: transparent !important;
    }

    div[data-testid="stExpander"] summary svg {
        fill: var(--ink) !important;
        color: var(--ink) !important;
    }

    div[data-testid="stExpander"] summary:hover,
    div[data-testid="stExpander"] summary:focus,
    div[data-testid="stExpander"] summary:active {
        background: rgba(15, 23, 42, 0.04) !important;
        color: var(--ink) !important;
    }

    div[data-testid="stAlert"] {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        color: var(--ink) !important;
    }

    div[data-testid="stAlert"] * {
        color: var(--ink) !important;
    }

    .progress-wrap {
        margin: 24px 0 6px;
    }

    .progress-step {
        display: grid;
        grid-template-columns: 54px 1fr auto;
        gap: 20px;
        align-items: center;
        padding: 16px 0;
        border-top: 1px solid var(--line);
        transition: opacity 0.25s ease;
    }

    .progress-step:last-child {
        border-bottom: 1px solid var(--line);
    }

    .progress-step.pending {
        opacity: 0.42;
    }

    .progress-num {
        font-size: 13px;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: 0.04em;
    }

    .progress-title {
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--ink);
    }

    .progress-desc {
        font-size: 13px;
        color: var(--muted);
        margin-top: 2px;
    }

    .progress-status {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--soft-muted);
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .progress-status::before {
        content: '';
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--soft-muted);
        opacity: 0.4;
    }

    .progress-step.done .progress-status {
        color: var(--success);
    }

    .progress-step.done .progress-status::before {
        background: var(--success);
        opacity: 1;
        box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.15);
    }

    .progress-step.running .progress-status {
        color: var(--warning);
    }

    .progress-step.running .progress-status::before {
        background: var(--warning);
        opacity: 1;
        animation: pl-pulse 1s ease-in-out infinite;
    }

    .progress-step.error .progress-status {
        color: var(--danger);
    }

    .progress-step.error .progress-status::before {
        background: var(--danger);
        opacity: 1;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.18);
    }

    .progress-step.skipped {
        opacity: 0.42;
    }

    .progress-step.skipped .progress-status::before {
        background: var(--soft-muted);
        opacity: 0.35;
    }

    .progress-error {
        margin-top: 10px;
        padding: 10px 14px;
        border-left: 3px solid var(--danger);
        background: var(--danger-bg);
        border-radius: 6px;
        font-size: 13px;
        color: var(--danger);
        line-height: 1.4;
        font-family: 'SFMono-Regular', Consolas, monospace;
        word-break: break-word;
    }

    @keyframes pl-pulse {
        0%, 100% {
            transform: scale(0.85);
            box-shadow: 0 0 0 0 rgba(202, 138, 4, 0.5);
        }
        50% {
            transform: scale(1.1);
            box-shadow: 0 0 0 6px rgba(202, 138, 4, 0);
        }
    }

    .footer-note {
        border-top: 1px solid var(--line);
        margin-top: 64px;
        padding-top: 18px;
        font-size: 12px;
        color: var(--soft-muted);
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .topbar {
            grid-template-columns: 1fr;
            gap: 8px;
            margin-bottom: 80px;
        }

        .topbar div {
            text-align: left !important;
        }

        .hero-title {
            font-size: 62px;
        }

        .hero-subtitle {
            font-size: 26px;
        }

        .agent-grid,
        .metrics-grid {
            grid-template-columns: 1fr;
        }

        .result-block {
            grid-template-columns: 1fr;
            gap: 8px;
        }
    }
</style>
""", unsafe_allow_html=True)


st.markdown(
"""
<div class="topbar">
<div class="brand-small">ProfileLab /</div>
<div>Optimizador multi-agente</div>
<div>Leonel Gordón</div>
</div>

<div class="hero-title">ProfileLab</div>
<div class="hero-subtitle">Desbloqueá tu posicionamiento profesional.</div>
<div class="hero-copy">
ProfileLab analiza un CV, interpreta el rol objetivo y genera una optimización de LinkedIn
mediante un flujo multi-agente con conocimiento externo recuperado por RAG.
</div>

<a class="hero-cta" href="#input-section">Optimizar mi perfil</a>

<div class="agent-grid">
<div class="agent-card">
<div class="agent-card-title">01<br>Parser</div>
<div class="agent-card-copy">Extrae y limpia el texto del CV cargado.</div>
</div>
<div class="agent-card">
<div class="agent-card-title">02<br>Profile Analyst</div>
<div class="agent-card-copy">Analiza seniority, foco, skills y keywords.</div>
</div>
<div class="agent-card">
<div class="agent-card-title">03<br>RAG Retriever</div>
<div class="agent-card-copy">Recupera patrones externos de optimización.</div>
</div>
<div class="agent-card">
<div class="agent-card-title">04<br>Strategist</div>
<div class="agent-card-copy">Genera headline, about, skills y recomendaciones.</div>
</div>
</div>

<div id="input-section" class="section">
<div class="section-title">Ejecutar optimización</div>
<div class="section-caption">Cargá un CV, definí el rol objetivo y seleccioná el provider.</div>
</div>
""",
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader(
    "CV",
    type=["pdf", "docx"],
    label_visibility="collapsed",
)

target_role = st.text_input(
    "Rol objetivo",
    placeholder="Ej: AI Engineer, Frontend Developer, Maestra, Product Manager",
)

provider = st.selectbox(
    "Provider",
    ["groq", "openai"],
)

run_button = st.button("Optimizar mi perfil")


if run_button:
    if not uploaded_file:
        st.error("Cargá un CV antes de ejecutar.")
    elif not target_role.strip():
        st.error("Ingresá un rol objetivo.")
    else:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            cv_file_path = tmp.name

        st.markdown(
'<div id="progress-anchor" class="scroll-anchor"></div>'
'<div class="result-section">'
'<div class="section-title">Progreso del análisis</div>'
'<div class="section-caption">Cada paso corresponde a un nodo del flujo multi-agente.</div>'
'</div>',
            unsafe_allow_html=True,
        )

        scroll_to_anchor("progress-anchor")

        progress_placeholder = st.empty()
        completed = set()
        errors = {}
        error_scrolled = {"value": False}
        render_progress(
            progress_placeholder,
            completed,
            PROGRESS_STEPS[0][0],
            errors,
        )

        def _on_node_done(node_name, node_error=None):
            completed.add(node_name)
            if node_error:
                # Mostramos un mensaje corto en el flujo; el detalle queda en trazabilidad.
                errors[node_name] = "Ocurrió un error. Revisá la trazabilidad del flujo."
                if not error_scrolled["value"]:
                    scroll_to_anchor("trace-logs-anchor", delay_ms=200)
                    error_scrolled["value"] = True
            remaining = [s[0] for s in PROGRESS_STEPS if s[0] not in completed]
            active = remaining[0] if remaining else None
            render_progress(progress_placeholder, completed, active, errors)

        try:
            result = run_profilelab_stream(
                cv_file_path=cv_file_path,
                target_role=target_role.strip(),
                provider=provider,
                on_node_done=_on_node_done,
            )
        except Exception as exc:  # noqa: BLE001
            remaining = [s[0] for s in PROGRESS_STEPS if s[0] not in completed]
            if remaining:
                errors[remaining[0]] = "Ocurrió un error. Revisá la trazabilidad del flujo."
                if not error_scrolled["value"]:
                    scroll_to_anchor("trace-logs-anchor", delay_ms=200)
                    error_scrolled["value"] = True
            render_progress(progress_placeholder, completed, None, errors)
            st.stop()

        render_progress(progress_placeholder, completed, None, errors)

        output = result.get("linkedin_output")
        logs = result.get("logs", [])
        chunks = result.get("retrieved_chunks", [])

        if output:
            headline = safe_text(getattr(output, "headline", ""))
            about = safe_text(getattr(output, "about", "")).replace("\n", "<br>")

            suggested_skills = getattr(output, "suggested_skills", []) or []
            seo_recommendations = getattr(output, "seo_recommendations", []) or []
            content_recommendations = getattr(output, "content_recommendations", []) or []

            skills_html = "".join(
                f"<div class='list-item'>• {safe_text(skill)}</div>"
                for skill in suggested_skills
            ) or "<div class='list-item'>—</div>"

            seo_html = "".join(
                f"<div class='list-item'>• {safe_text(item)}</div>"
                for item in seo_recommendations
            ) or "<div class='list-item'>—</div>"

            content_html = "".join(
                f"<div class='list-item'>• {safe_text(item)}</div>"
                for item in content_recommendations
            ) or "<div class='list-item'>—</div>"

            st.markdown(
'<div id="result-anchor" class="scroll-anchor"></div>'
'<div class="result-section">'
'<div class="section-title">Resultado generado</div>'
'<div class="section-caption">Resultado generado a partir del análisis multi-agente del perfil y contexto externo.</div>'
'</div>',
                unsafe_allow_html=True,
            )

            scroll_to_anchor("result-anchor", delay_ms=400)

            st.markdown(
f'<div class="result-block">'
f'<div class="result-label">Headline</div>'
f'<div class="result-value">{headline}</div>'
f'</div>'
f'<div class="result-block">'
f'<div class="result-label">About</div>'
f'<div class="result-value">{about}</div>'
f'</div>'
f'<div class="result-block">'
f'<div class="result-label">Skills sugeridas</div>'
f'<div class="result-value">{skills_html}</div>'
f'</div>'
f'<div class="result-block">'
f'<div class="result-label">Recomendaciones SEO</div>'
f'<div class="result-value">{seo_html}</div>'
f'</div>'
f'<div class="result-block">'
f'<div class="result-label">Ideas de contenido</div>'
f'<div class="result-value">{content_html}</div>'
f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
'<div id="trace-anchor" class="scroll-anchor"></div>'
'<div class="result-section">'
'<div class="section-title">Métricas y trazabilidad</div>'
'<div class="section-caption">Información técnica usada para validar ejecución, observabilidad y recuperación de contexto.</div>'
'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
f'<div class="metrics-grid">'
f'<div class="metric-card">'
f'<div class="metric-label">Provider</div>'
f'<div class="metric-value">{safe_text(provider)}</div>'
f'</div>'
f'<div class="metric-card">'
f'<div class="metric-label">Rol objetivo</div>'
f'<div class="metric-value">{safe_text(target_role.strip())}</div>'
f'</div>'
f'<div class="metric-card">'
f'<div class="metric-label">Chunks RAG</div>'
f'<div class="metric-value">{len(chunks)}</div>'
f'</div>'
f'<div class="metric-card">'
f'<div class="metric-label">Logs</div>'
f'<div class="metric-value">{len(logs)}</div>'
f'</div>'
f'</div>'
'<div class="section-gap"></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div id="trace-logs-anchor" class="scroll-anchor"></div>', unsafe_allow_html=True)

        with st.expander("Ver trazabilidad del flujo", expanded=bool(errors)):
            if logs:
                for visible_line in _render_visible_log_lines(logs):
                    st.write(visible_line)
            else:
                st.write("No se registraron logs.")

        with st.expander("Ver contexto RAG utilizado"):
            if chunks:
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(chunk)
            else:
                st.write("No se recuperó contexto RAG.")


st.markdown(
    """
    <div class="footer-note">
        ProfileLab — POC multi-agente para optimización profesional.<br>
        Diseñado con LangGraph, RAG y salida estructurada.<br><br>
        Creado por <strong>Leonel Gordón</strong>
    </div>
    """,
    unsafe_allow_html=True,
)