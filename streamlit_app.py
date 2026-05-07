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

REFINE_PROGRESS_STEPS = [
    ("output_refiner", "01", "Refinar output", "Aplicando tu feedback con el agente de refinamiento."),
]


def render_progress(placeholder, completed, active, errors=None, steps=None):
    errors = errors or {}
    steps = steps or PROGRESS_STEPS
    first_error_seen = False
    rows = []

    for key, num, title, desc in steps:
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


def run_graph_stream(initial_state: dict, on_node_done):
    graph = build_graph()
    accumulated = dict(initial_state)

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            node_error = None
            if isinstance(update, dict):
                accumulated.update(update)
                node_error = update.get("error")
            on_node_done(node_name, node_error)

    return accumulated


def run_profilelab_stream(cv_file_path: str, target_role: str, provider: str, on_node_done):
    initial_state = {
        "cv_file_path": cv_file_path,
        "target_role": target_role,
        "provider": provider,
        "logs": [],
    }
    return run_graph_stream(initial_state, on_node_done)


def build_linkedin_output_blocks_html(output) -> str:
    """Genera el HTML de bloques de resultado; acepta LinkedInOutput o dict."""
    if output is None:
        return ""

    if hasattr(output, "headline"):
        headline = safe_text(getattr(output, "headline", ""))
        about = safe_text(getattr(output, "about", "")).replace("\n", "<br>")
        suggested_skills = getattr(output, "suggested_skills", []) or []
        seo_recommendations = getattr(output, "seo_recommendations", []) or []
        content_recommendations = getattr(output, "content_recommendations", []) or []
    else:
        headline = safe_text(output.get("headline", ""))
        about = safe_text(output.get("about", "")).replace("\n", "<br>")
        suggested_skills = output.get("suggested_skills") or []
        seo_recommendations = output.get("seo_recommendations") or []
        content_recommendations = output.get("content_recommendations") or []

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

    return (
        f'<div class="result-block">'
        f'<div class="result-label">Headline</div>'
        f'<div class="result-value">{headline}</div>'
        f"</div>"
        f'<div class="result-block">'
        f'<div class="result-label">About</div>'
        f'<div class="result-value">{about}</div>'
        f"</div>"
        f'<div class="result-block">'
        f'<div class="result-label">Skills sugeridas</div>'
        f'<div class="result-value">{skills_html}</div>'
        f"</div>"
        f'<div class="result-block">'
        f'<div class="result-label">Recomendaciones SEO</div>'
        f'<div class="result-value">{seo_html}</div>'
        f"</div>"
        f'<div class="result-block">'
        f'<div class="result-label">Ideas de contenido</div>'
        f'<div class="result-value">{content_html}</div>'
        f"</div>"
    )


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
        st.session_state.pop("pl_refined_output", None)
        st.session_state["pl_iteration_history"] = []

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
            steps=PROGRESS_STEPS,
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
            render_progress(
                progress_placeholder, completed, active, errors, steps=PROGRESS_STEPS
            )

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
            render_progress(
                progress_placeholder, completed, None, errors, steps=PROGRESS_STEPS
            )
            st.stop()

        render_progress(progress_placeholder, completed, None, errors, steps=PROGRESS_STEPS)

        if result.get("error"):
            st.error(
                "El pipeline terminó con error. Revisá la trazabilidad más abajo si hay resultados parciales."
            )

        # Persistimos el estado de progreso para que no desaparezca en reruns
        # (por ejemplo, cuando el usuario refina el output).
        st.session_state["pl_last_progress"] = {
            "completed": sorted(list(completed)),
            "errors": dict(errors),
        }

        if result.get("linkedin_output") and not result.get("error"):
            st.session_state["pl_linkedin_output"] = result.get("linkedin_output")
            st.session_state["pl_refined_output"] = result.get("refined_output")
            st.session_state["pl_profile_analysis"] = result.get("profile_analysis")
            st.session_state["pl_target_role"] = target_role.strip()
            st.session_state["pl_provider"] = provider
            st.session_state["pl_iteration_history"] = list(
                result.get("iteration_history") or []
            )
            st.session_state["pl_logs"] = result.get("logs", [])
            st.session_state["pl_chunks"] = result.get("retrieved_chunks", [])

if st.session_state.get("pl_last_progress") and not run_button:
    last = st.session_state["pl_last_progress"] or {}
    st.markdown(
        '<div class="result-section">'
        '<div class="section-title">Progreso (última ejecución)</div>'
        '<div class="section-caption">Se mantiene visible mientras refinás el output.</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    prog_placeholder = st.empty()
    render_progress(
        prog_placeholder,
        set(last.get("completed") or []),
        None,
        last.get("errors") or {},
        steps=PROGRESS_STEPS,
    )

if st.session_state.get("pl_linkedin_output"):
    pl_provider = st.session_state.get("pl_provider", provider)
    pl_target = st.session_state.get("pl_target_role", target_role or "")
    pl_logs = st.session_state.get("pl_logs", [])
    pl_chunks = st.session_state.get("pl_chunks", [])
    linkedin_initial = st.session_state["pl_linkedin_output"]
    refined = st.session_state.get("pl_refined_output")
    current_output = refined if refined is not None else linkedin_initial
    hist = st.session_state.get("pl_iteration_history") or []

    st.markdown(
        '<div id="result-anchor" class="scroll-anchor"></div>'
        '<div class="result-section">'
        '<div class="section-title">Resultado actual</div>'
        '<div class="section-caption">Versión vigente (refinada si aplicaste feedback). Generada a partir del análisis multi-agente y contexto RAG.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.pop("pl_scroll_to_result", False):
        scroll_to_anchor("result-anchor", delay_ms=450)

    if st.session_state.pop("pl_refine_success", False):
        st.success("Listo: refinamiento aplicado.")

    st.markdown(
        build_linkedin_output_blocks_html(current_output),
        unsafe_allow_html=True,
    )

    with st.expander("Historial de versiones", expanded=False):
        st.caption("Versión inicial: salida del Strategist. Las siguientes entradas son refinamientos.")
        with st.expander("Versión inicial (Strategist)", expanded=False):
            st.markdown(
                build_linkedin_output_blocks_html(linkedin_initial),
                unsafe_allow_html=True,
            )
        for i, entry in enumerate(hist, start=1):
            fb_full = entry.get("user_feedback") or ""
            fb_preview = fb_full[:100]
            label = f"Versión {i} (refinado)"
            if fb_preview:
                suffix = "…" if len(fb_full) > 100 else ""
                label += f" — {fb_preview}{suffix}"
            with st.expander(label, expanded=False):
                st.markdown(
                    build_linkedin_output_blocks_html(entry.get("refined_output")),
                    unsafe_allow_html=True,
                )

    st.markdown(
        '<div class="result-section">'
        '<div class="section-title">Refinar con feedback</div>'
        '<div class="section-caption">Describí cambios concretos (tono, menos técnico, más keywords, etc.). Se ejecuta solo el nodo de refinamiento, sin re-procesar el CV.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Streamlit no permite modificar session_state de un widget luego de instanciarlo.
    # Usamos un flag para limpiar el textarea en el siguiente rerun.
    if st.session_state.pop("pl_clear_feedback", False):
        st.session_state["pl_feedback_text"] = ""

    last_refine = st.session_state.get("pl_last_refine_progress") or None
    if last_refine:
        with st.expander("Último refinamiento (estado)", expanded=False):
            rp = st.empty()
            render_progress(
                rp,
                set(last_refine.get("completed") or []),
                None,
                last_refine.get("errors") or {},
                steps=REFINE_PROGRESS_STEPS,
            )

    feedback_text = st.text_area(
        "Feedback",
        key="pl_feedback_text",
        height=120,
        placeholder="Ej.: Hacé el about menos técnico y más orientado a impacto de negocio.",
    )

    refine_clicked = st.button("Aplicar feedback", type="primary", key="pl_refine_btn")

    if refine_clicked:
        if not st.session_state.get("pl_profile_analysis"):
            st.error("No hay análisis de perfil en sesión. Volvé a ejecutar «Optimizar mi perfil».")
        elif not (feedback_text or "").strip():
            st.warning("Escribí feedback antes de aplicar.")
        else:
            refine_placeholder = st.empty()
            r_completed = set()
            r_errors = {}
            r_error_scrolled = {"value": False}
            render_progress(
                refine_placeholder,
                r_completed,
                REFINE_PROGRESS_STEPS[0][0],
                r_errors,
                steps=REFINE_PROGRESS_STEPS,
            )

            def _on_refine_done(node_name, node_error=None):
                r_completed.add(node_name)
                if node_error:
                    r_errors[node_name] = (
                        "Ocurrió un error. Revisá la trazabilidad del flujo."
                    )
                    if not r_error_scrolled["value"]:
                        scroll_to_anchor("trace-logs-anchor", delay_ms=200)
                        r_error_scrolled["value"] = True
                remaining = [
                    s[0] for s in REFINE_PROGRESS_STEPS if s[0] not in r_completed
                ]
                active = remaining[0] if remaining else None
                render_progress(
                    refine_placeholder,
                    r_completed,
                    active,
                    r_errors,
                    steps=REFINE_PROGRESS_STEPS,
                )

            refine_state = {
                "refine_only": True,
                "user_feedback": feedback_text.strip(),
                "linkedin_output": st.session_state["pl_linkedin_output"],
                "refined_output": st.session_state.get("pl_refined_output"),
                "profile_analysis": st.session_state["pl_profile_analysis"],
                "target_role": (target_role or "").strip() or pl_target,
                "provider": provider,
                "logs": list(st.session_state.get("pl_logs", [])),
                "iteration_history": list(
                    st.session_state.get("pl_iteration_history", [])
                ),
            }

            try:
                refine_result = run_graph_stream(refine_state, _on_refine_done)
            except Exception as exc:  # noqa: BLE001
                r_errors[REFINE_PROGRESS_STEPS[0][0]] = (
                    "Ocurrió un error. Revisá la trazabilidad del flujo."
                )
                render_progress(
                    refine_placeholder,
                    r_completed,
                    None,
                    r_errors,
                    steps=REFINE_PROGRESS_STEPS,
                )
                st.error(str(exc))
            else:
                render_progress(
                    refine_placeholder,
                    r_completed,
                    None,
                    r_errors,
                    steps=REFINE_PROGRESS_STEPS,
                )
                if refine_result.get("error"):
                    st.error(
                        "No se pudo refinar el output. Revisá la trazabilidad del flujo."
                    )
                    st.session_state["pl_last_refine_progress"] = {
                        "completed": sorted(list(r_completed)),
                        "errors": dict(r_errors),
                    }
                else:
                    st.session_state["pl_refined_output"] = refine_result.get(
                        "refined_output"
                    )
                    st.session_state["pl_iteration_history"] = list(
                        refine_result.get("iteration_history") or []
                    )
                    st.session_state["pl_logs"] = refine_result.get("logs", [])
                    st.session_state["pl_last_refine_progress"] = {
                        "completed": ["output_refiner"],
                        "errors": {},
                    }
                    # Limpiamos el textarea y forzamos rerun para que el "Resultado actual"
                    # (que se renderiza arriba) se actualice inmediatamente.
                    st.session_state["pl_clear_feedback"] = True
                    st.session_state["pl_scroll_to_result"] = True
                    st.session_state["pl_refine_success"] = True
                    st.rerun()

    st.markdown(
        '<div id="trace-anchor" class="scroll-anchor"></div>'
        '<div class="result-section">'
        '<div class="section-title">Métricas y trazabilidad</div>'
        '<div class="section-caption">Información técnica usada para validar ejecución, observabilidad y recuperación de contexto.</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="metrics-grid">'
        f'<div class="metric-card">'
        f'<div class="metric-label">Provider</div>'
        f'<div class="metric-value">{safe_text(pl_provider)}</div>'
        f"</div>"
        f'<div class="metric-card">'
        f'<div class="metric-label">Rol objetivo</div>'
        f'<div class="metric-value">{safe_text(pl_target)}</div>'
        f"</div>"
        f'<div class="metric-card">'
        f'<div class="metric-label">Chunks RAG</div>'
        f'<div class="metric-value">{len(pl_chunks)}</div>'
        f"</div>"
        f'<div class="metric-card">'
        f'<div class="metric-label">Logs</div>'
        f'<div class="metric-value">{len(pl_logs)}</div>'
        f"</div>"
        f"</div>"
        '<div class="section-gap"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div id="trace-logs-anchor" class="scroll-anchor"></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Ver trazabilidad del flujo", expanded=False):
        if pl_logs:
            for visible_line in _render_visible_log_lines(pl_logs):
                st.write(visible_line)
        else:
            st.write("No se registraron logs.")

    with st.expander("Ver contexto RAG utilizado"):
        if pl_chunks:
            for i, chunk in enumerate(pl_chunks, start=1):
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