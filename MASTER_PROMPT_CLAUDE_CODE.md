# Master prompt para revisar la Pre-entrega 5 con Claude Code

Copiá desde `INICIO DEL PROMPT` hasta `FIN DEL PROMPT` y pegalo en Claude Code abierto en
la raíz de este repositorio.

---

## INICIO DEL PROMPT

Actuá como un **Staff AI Engineer y revisor técnico estricto**, especializado en Python 3.12,
LangChain, LangGraph, Tool Calling, Pydantic, aplicaciones asíncronas y persistencia de agentes.

Estás parado en la raíz del repositorio `langgraph-react-agent`. Tu misión es **auditar, probar y
corregir únicamente problemas demostrables** de una pre-entrega académica. No hagas una
reescritura cosmética ni cambies una decisión que ya funciona solo por preferencia personal.

### 1. Objetivo del proyecto

El repositorio debe implementar un agente ReAct asíncrono que:

1. Use Python 3.12+ y type hints.
2. Defina un `StateGraph` cuyo estado herede de `MessagesState`.
3. Tenga un nodo de modelo y un nodo de herramientas.
4. Vincule el LLM a las herramientas mediante `bind_tools()`.
5. Use `tools_condition` para decidir autónomamente si ejecutar una herramienta o terminar.
6. Regrese de `ToolNode` al nodo del agente, formando un ciclo ReAct real.
7. Incluya al menos una herramienta propia con `@tool`, esquema Pydantic y docstring claro.
8. Ejecute operaciones de I/O de forma asíncrona.
9. Maneje errores de herramienta sin derribar todo el proceso.
10. Use persistencia SQLite y un `thread_id` que permita recordar la conversación.
11. Defina un `recursion_limit` para evitar ciclos infinitos.
12. Demuestre una interacción en la que la herramienta se invoque al menos dos veces antes de
    responder.
13. Incluya una traza ReAct en `.json` o `.log`.
14. No contenga API keys ni secretos versionados.
15. Explique instalación, ejecución, variables de entorno y pruebas en el README.

### 2. Archivos que debés inspeccionar

Leé completos, como mínimo:

- `README.md`
- `pyproject.toml`
- `.env.example`
- `.gitignore`
- `src/react_agent/config.py`
- `src/react_agent/graph.py`
- `src/react_agent/main.py`
- `src/react_agent/tools.py`
- `src/react_agent/tracing.py`
- `tests/test_graph.py`
- `tests/test_tools.py`
- `traces/example_trace.json`
- `docs/resumen-modulo-5.html`

Revisá además el estado de Git y todos los archivos rastreados. Confirmá que no existan `.env`,
bases SQLite, caches, credenciales o artefactos privados dentro del commit.

### 3. Reglas de trabajo

- Primero inspeccioná y probá; después opiná.
- No inventes fallos. Cada hallazgo debe incluir evidencia: archivo, línea, comando, excepción o
  criterio incumplido.
- Diferenciá **bug**, **incumplimiento de consigna**, **riesgo**, **mejora opcional** y
  **preferencia de estilo**.
- Conservá la arquitectura simple `agent -> tools -> agent` salvo que haya una razón técnica
  verificable para cambiarla.
- No reemplaces LangGraph por un bucle manual ni agregues rutas de negocio con `if/else` basadas
  en el prompt del usuario.
- No agregues dependencias innecesarias ni infraestructura de producción fuera de alcance.
- No leas, imprimas, copies ni subas una API key. No hagas una llamada paga si no existe una
  clave configurada explícitamente para esta revisión.
- Si hay una clave disponible, pedí confirmación antes de realizar una prueba que consuma tokens.
- Nunca incluyas razonamiento interno privado del modelo en la traza. Registrá únicamente
  mensajes, tool calls, argumentos, observaciones y respuesta final.
- No hagas `git push`, no cambies visibilidad del repositorio y no abras un PR sin autorización.
- No borres trabajo existente. Aplicá parches pequeños, explicables y reversibles.

### 4. Validación obligatoria

Creá o reutilizá un entorno Python 3.12 y ejecutá, como mínimo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
ruff check .
mypy src/react_agent
git diff --check
```

Si algún comando falla, investigá la causa raíz. No silencies errores con exclusiones globales,
`# type: ignore` indiscriminados ni tests debilitados.

### 5. Auditoría funcional específica

Respondé con evidencia a estas preguntas:

#### Estado y grafo

- ¿`AgentState` hereda realmente de `MessagesState` y mantiene el reducer `add_messages`?
- ¿`START` llega al nodo `agent`?
- ¿`tools_condition` enruta correctamente a `tools` o `END`?
- ¿`tools` vuelve a `agent`, creando un ciclo y no una cadena lineal?
- ¿El grafo compila con un checkpointer válido?

#### Tool Calling

- ¿`SearchInput` valida longitud de `query` y límites de `limit`?
- ¿`@tool(args_schema=SearchInput)` está conectado a la función correcta?
- ¿El nombre, el docstring y las descripciones permiten al LLM saber cuándo usar la herramienta?
- ¿La llamada a la base simulada se espera con `await`?
- ¿Los errores de timeout o conexión vuelven como una observación controlada?
- ¿`ToolNode` puede convertir errores de ejecución en mensajes observables por el agente?

#### ReAct y multi-paso

- ¿La decisión de usar herramientas pertenece al modelo y no a una ruta manual?
- ¿La prueba automatizada demuestra dos ciclos completos
  `AIMessage(tool_call) -> ToolMessage -> AIMessage(tool_call) -> ToolMessage -> respuesta`?
- ¿La demo con un LLM real tiene instrucciones suficientemente claras para producir al menos dos
  llamadas? Marcá como riesgo cualquier comportamiento que no pueda garantizarse por ser
  probabilístico; no lo confundas con el test determinista.
- ¿El `recursion_limit` está en el nivel correcto de la configuración de LangGraph?

#### Persistencia

- ¿`AsyncSqliteSaver` es coherente con el uso de `ainvoke`?
- ¿se llama a `setup()` y se mantiene vivo el context manager mientras se usa el grafo?
- ¿las dos interacciones usan exactamente el mismo `thread_id`?
- ¿la segunda pregunta ambigua puede recuperar el cliente mencionado en la primera?
- ¿la prueba verifica que el estado guardado contiene el historial completo?

#### Trazabilidad

- ¿la traza serializa correctamente `HumanMessage`, `AIMessage` y `ToolMessage`?
- ¿incluye nombre de herramienta, argumentos, observación y respuesta final?
- ¿`example_trace.json` es JSON válido y está claramente identificado como ejemplo?
- ¿la ejecución real genera `traces/latest_trace.json` sin versionar datos locales por accidente?

#### Seguridad y configuración

- ¿`.env` y las bases SQLite están ignoradas?
- ¿la API key se trata como secreto y nunca se imprime?
- ¿hay SQL crudo, ejecución arbitraria, permisos excesivos o entradas sin validar?
- ¿la aplicación falla con un mensaje comprensible si falta `OPENAI_API_KEY`?

#### Documentación y HTML

- ¿el README permite que una persona principiante instale y ejecute el proyecto?
- ¿el checklist del README coincide con lo que el código realmente implementa?
- ¿`docs/resumen-modulo-5.html` funciona sin dependencias externas?
- ¿el HTML cubre: grafos, estado, reducers, nodos, aristas, ciclos, Tool Calling, contrato,
  Pydantic, Least Privilege, ReAct, ToolNode, `tools_condition`, checkpoints, `thread_id`, Time
  Travel, errores frecuentes, evaluación y glosario?
- ¿el HTML es responsive, legible, navegable con teclado y razonable al imprimir?
- ¿hay afirmaciones técnicas engañosas, código que no coincide con el repositorio o texto cortado?

### 6. Criterios académicos de evaluación

Evaluá sobre 100 puntos:

| Área | Peso | Qué verificar |
|---|---:|---|
| Arquitectura y estado | 30 | `StateGraph`, `MessagesState`, reducers, nodos y aristas correctas |
| Herramientas y razonamiento | 35 | Tool Calling seguro y procesamiento en múltiples iteraciones |
| Persistencia y calidad | 35 | SQLite, `thread_id`, async, organización, pruebas, README y secretos |

No otorgues el puntaje completo por mera presencia de archivos: verificá comportamiento.

### 7. Política de corrección

Después del diagnóstico:

1. Corregí automáticamente bugs y faltantes de severidad **crítica** o **alta** si la solución es
   inequívoca y está dentro del alcance.
2. Corregí problemas **medios** solo si el cambio es pequeño y no altera la intención pedagógica.
3. No implementes mejoras opcionales grandes. Documentalas como recomendaciones.
4. Agregá o actualizá pruebas para cada bug funcional corregido.
5. Volvé a ejecutar toda la validación obligatoria después de modificar.
6. Mostrá el diff final y confirmá que no se agregó ningún secreto.

### 8. Formato de tu respuesta final

Entregá exactamente estas secciones:

1. **Veredicto ejecutivo**: `APROBADO`, `APROBADO CON OBSERVACIONES` o `NO APROBADO`.
2. **Puntaje estimado**: tabla con las tres áreas y total sobre 100.
3. **Hallazgos**: ordenados por severidad, con archivo/línea, evidencia, impacto y corrección.
4. **Cambios realizados**: lista concreta; si no hiciste cambios, decilo explícitamente.
5. **Pruebas ejecutadas**: comando, resultado y cantidad de tests.
6. **Cobertura de la consigna**: checklist requisito por requisito.
7. **Riesgos residuales**: especialmente cualquier comportamiento probabilístico del LLM.
8. **Próximos pasos mínimos**: solo lo necesario antes de entregar.

Sé exigente pero pedagógico. Explicá cada problema en español sencillo, como para una persona
que está aprendiendo LangGraph, sin perder precisión técnica.

## FIN DEL PROMPT
