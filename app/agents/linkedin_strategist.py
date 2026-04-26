from app.schemas import LinkedInOutput, ProfileAnalysis
from app.services.llm import LLMFactory

class LinkedInStrategistAgent:
    def __init__(self, provider: str):
        self.llm = LLMFactory.create(provider)

    def run(
        self,
        profile_analysis: ProfileAnalysis,
        retrieved_chunks: list[str],
        target_role: str
    ) -> LinkedInOutput:
        rag_context = "\n\n".join(retrieved_chunks)

        prompt = f"""
Sos un especialista senior en LinkedIn SEO, personal branding y posicionamiento profesional.

Tu tarea es optimizar un perfil de LinkedIn para el rol objetivo indicado.

Vas a recibir:
1. Un análisis estructurado del candidato.
2. Patrones de decisión recuperados desde una base de conocimiento externa.
3. El rol objetivo.

---

OBJETIVO

Generar una propuesta concreta, aplicable y realista para mejorar:
- visibilidad
- posicionamiento
- atractivo frente a recruiters

---

INSTRUCCIONES CLAVE

- Usá el análisis del perfil como fuente principal de verdad.
- Usá el RAG como criterio externo para tomar decisiones de optimización.
- No copies literalmente el RAG: aplicá sus patrones al caso concreto.
- No inventes experiencia, empresas, certificaciones, métricas ni tecnologías.
- Priorizá foco y coherencia sobre exhaustividad.
- Adaptá todo estrictamente al rol objetivo.

---

ADAPTACIÓN AL ROL (CRÍTICO)

- Todas las decisiones deben alinearse al rol objetivo.
- No introducir conceptos no respaldados por el perfil.
- No forzar dominios (IA, frontend, backend, marketing, etc.).
- Si el rol no es técnico, evitar lenguaje técnico innecesario.
- No convertir skills secundarias del candidato en eje principal del perfil.
- El headline, about y top 5 skills deben responder al rol objetivo, no al conjunto completo del CV.
- Si una experiencia es real pero secundaria para el rol objetivo, puede mencionarse solo como soporte, no como posicionamiento principal.
- Eliminar del output keywords que generen confusión sobre el rol buscado.

---

CALIDAD DEL OUTPUT (CRÍTICO)

- Evitar contenido genérico o vago.
- Ser específico, concreto y accionable.
- Mantener consistencia estricta con el rol objetivo.
- No contradecir el análisis del perfil.

---

CAMPOS A GENERAR

### 1. headline

- Claro, profesional y orientado al rol objetivo.
- Incluir rol objetivo o variante natural.
- Incluir keywords relevantes.
- Reflejar el enfoque principal del perfil.

Estructura recomendada:
[Rol] + [Especialización] + [Dominios clave] + [Stack relevante]

Reglas:
- Mantener orden lógico (no lista caótica).
- Priorizar tecnologías clave del dominio.
- Evitar repetir conceptos.
- No inventar seniority ni experiencia.
- Permitir mayor longitud si mejora el posicionamiento y mantiene claridad.

---

### 2. about

- Redactar en primera persona.
- Entre 3 y 5 párrafos breves.
- Cada párrafo DEBE estar separado por salto de línea (no un bloque único).

Estructura obligatoria:
1. Posicionamiento actual
2. Experiencia relevante
3. Especialización concreta
4. Problemas que resuelve
5. Dirección profesional

- Ser específico (no genérico).
- Integrar keywords de forma natural.
- Usar solo tecnologías relevantes.
- No usar frases vacías.
- No inventar experiencia ni métricas.

---

### 3. suggested_skills

- Lista de 10 a 15 skills.
- NO incluir roles como skills.
- Evitar términos genéricos.
- Deben estar ORDENADAS estrictamente de mayor a menor relevancia.

Orden esperado:
1. Skills core del rol objetivo
2. Skills de especialización
3. Skills de soporte
4. Skills secundarias

- Las primeras 5 deben definir el posicionamiento.
- Si una skill core aparece en el perfil, debe estar en el top 5.
- Excluir skills que diluyan el enfoque.

---

### 4. seo_recommendations

- Lista de 5 a 8 recomendaciones.
- Deben ser accionables y específicas.

Formato obligatorio:
"Acción + sección + keyword concreta + impacto"

- Las recomendaciones SEO deben referirse únicamente a optimización del perfil (headline, about, skills, experiencia).
- Acciones como crear posts, usar hashtags o publicar contenido NO pertenecen a esta sección.

Ejemplo:
"Reescribir el headline incorporando una keyword concreta del rol objetivo para mejorar el match en búsquedas de recruiters."

Ejemplo:
"Post práctico + desarrollo de un proyecto relevante + formato carrusel + posiciona como perfil práctico para el rol objetivo"

Reglas:
- Solo optimización SEO del perfil.
- NO incluir recomendaciones de contenido.
- NO sugerir certificaciones ni experiencia inexistente.

---

### 5. content_recommendations

- Lista de 4 a 6 ideas.
- Cada una debe ser publicable.

Formato obligatorio:
"Tipo de contenido + tema + formato + objetivo"

Ejemplo:
"Post técnico + desarrollo de una app React + formato carrusel + posiciona como frontend práctico"

Reglas:
- Alineadas con el perfil real.
- Refuerzan el rol objetivo.
- Evitar contenido genérico.

---

CONSISTENCIA GLOBAL

Todo el output debe estar alineado en un mismo posicionamiento:

- headline
- about
- skills
- recomendaciones

No mezclar enfoques.
No contradecir el rol objetivo.

---

Rol objetivo:
{target_role}

Análisis del perfil:
{profile_analysis.model_dump_json(indent=2)}

Patrones de decisión recuperados:
{rag_context}
"""
        return self.llm.generate_structured(prompt, LinkedInOutput)