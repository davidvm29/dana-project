import os
import json
import re
import duckdb
import pandas as pd

from dash import Dash, html, dcc, Input, Output, State, no_update
from dash import dash_table
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_extensions.javascript import assign

import plotly.express as px

# ============================================================
# FLAGS / PATHS
# ============================================================

ENABLE_ASSISTANT = os.getenv("ENABLE_ASSISTANT", "0") == "1"

DB_PATH = os.getenv("DANA_DB_PATH", "data/db/dana.duckdb")
GEOJSON_PATH = os.getenv("DANA_GEOJSON_PATH", "cv_municipios.geojson")
PUNTOS_PARQUET_PATH = os.getenv("DANA_PUNTOS_PATH", "data/puntos_filtrados.parquet")

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]
)

server = app.server

PAGE_STYLE = {
    "backgroundColor": "#f5f7fa",
    "paddingBottom": "40px"
}

CARD_CLASS = "shadow-sm mb-4"

# ============================================================
# 1. CARGA DE DATOS DESDE DUCKDB
# ============================================================

def load_duckdb_tables(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(db_path, read_only=True)
    df_local = con.execute("SELECT * FROM datos_procesados").fetchdf()
    df_muni_local = con.execute("SELECT * FROM datos_municipio").fetchdf()
    con.close()
    return df_local, df_muni_local

df, df_muni = load_duckdb_tables(DB_PATH)
print("Datos cargados desde DuckDB")

# ============================================================
# 2. GEOJSON — FILTRADO A MUNICIPIOS DISPONIBLES
# ============================================================

municipios_validos = set(df_muni["nombre_norm"])
df_dict = df_muni.set_index("nombre_norm").to_dict(orient="index")

with open(GEOJSON_PATH, encoding="utf-8") as f:
    geojson_cv = json.load(f)

features = []

for feat in geojson_cv["features"]:
    nombre = feat["properties"].get("municipio")
    if nombre not in municipios_validos:
        continue

    datos = df_dict[nombre]

    feat["properties"]["tooltip"] = nombre
    feat["properties"]["popup"] = (
        f"<b>{nombre}</b><br>"
        f"Edificios inspeccionados: {datos['edificios']}<br>"
        f"IGD medio: {round(datos['igd_medio'], 2)}<br>"
        f"IGD máximo: {round(datos['igd_max'], 2)}<br>"
        f"Daño medio: {round(datos['danio_medio'], 2)}<br>"
        f"Daño máximo: {round(datos['danio_max'], 2)}<br>"
        f"Altura media de agua: {round(datos['altura_media_agua'], 2)} m<br>"
        f"Altura máxima de agua: {round(datos['altura_max_agua'], 2)} m"
    )

    # valores numéricos para la coropleta
    feat["properties"]["igd_medio"] = float(datos["igd_medio"])
    feat["properties"]["igd_max"] = float(datos["igd_max"])
    feat["properties"]["danio_medio"] = float(datos["danio_medio"])
    feat["properties"]["danio_max"] = float(datos["danio_max"])
    feat["properties"]["edificios"] = int(datos["edificios"])
    feat["properties"]["altura_media_agua"] = float(datos["altura_media_agua"])
    feat["properties"]["altura_max_agua"] = float(datos["altura_max_agua"])

    features.append(feat)

geojson_filtrado = {"type": "FeatureCollection", "features": features}

# ============================================================
# 3. CAPA DE PUNTOS — CARGA DESDE PARQUET PRECALCULADO
# ============================================================
# Este parquet lo generas en local con un script (precompute_points.py)
# para evitar geopandas en runtime (más estable en Vercel).
# Debe contener al menos estas columnas:
# lat, lon, nombremunicipio, igd, danos_total, cotaagua_normalizada, imagencatastro
# ============================================================

if not os.path.exists(PUNTOS_PARQUET_PATH):
    raise FileNotFoundError(
        f"No existe {PUNTOS_PARQUET_PATH}. "
        "Genera el parquet de puntos filtrados en local y súbelo al repo."
    )

df_puntos = pd.read_parquet(PUNTOS_PARQUET_PATH)

# Validación mínima de columnas esperadas
required_cols = {"lat", "lon", "nombremunicipio", "igd", "danos_total", "cotaagua_normalizada", "imagencatastro"}
missing = required_cols - set(df_puntos.columns)
if missing:
    raise ValueError(f"Al parquet de puntos le faltan columnas: {sorted(missing)}")

# ============================================================
# 4. MARKERS (PUNTOS) — POPUP CON IMAGEN
# ============================================================

markers = [
    dl.CircleMarker(
        center=[row.lat, row.lon],
        radius=2,
        color="#1f78b4",
        fill=True,
        fillColor="#1f78b4",
        fillOpacity=0.35,
        children=[
            dl.Tooltip(f"{row.nombremunicipio}"),
            dl.Popup([
                html.B(row.nombremunicipio),
                html.Br(),
                f"IGD: {row.igd}",
                html.Br(),
                f"Daño total: {row.danos_total}",
                html.Br(),
                f"Cota agua: {row.cotaagua_normalizada}",
                html.Br(),
                html.Img(
                    src=row.imagencatastro,
                    style={"width": "240px", "border": "1px solid #ddd", "borderRadius": "6px"}
                )
            ])
        ]
    )
    for row in df_puntos.itertuples(index=False)
]

# ============================================================
# 5. STYLEFUNCTION GEOJSON
# ============================================================

geojson_style = assign("""
function(feature, context){
    const variable = context.hideout.variable;
    const valor = feature.properties[variable];
    let color = "#cccccc";

    // IGD escala azul
    if (variable === "igd_medio" || variable === "igd_max") {
        if (valor < 1) color = "#d4eeff";
        else if (valor < 4) color = "#74c0e3";
        else if (valor < 10) color = "#1f78b4";
        else color = "#08306b";
    }
    // Daños
    else if (variable === "danio_medio") {
        if (valor < 0.5) color = "#fee5d9";
        else if (valor < 1.0) color = "#fcae91";
        else if (valor < 2.0) color = "#fb6a4a";
        else color = "#cb181d";
    }
    else if (variable === "danio_max") {
        if (valor < 2) color = "#fee5d9";
        else if (valor < 5) color = "#fcae91";
        else if (valor < 10) color = "#fb6a4a";
        else color = "#cb181d";
    }
    // Edificios
    else if (variable === "edificios") {
        if (valor < 50) color = "#fee5d9";
        else if (valor < 150) color = "#fcae91";
        else if (valor < 400) color = "#fb6a4a";
        else color = "#cb181d";
    }
    // Altura agua
    else if (variable === "altura_media_agua" || variable === "altura_max_agua") {
        if (valor < 0.2) color = "#fee5d9";
        else if (valor < 0.6) color = "#fcae91";
        else if (valor < 1.2) color = "#fb6a4a";
        else color = "#cb181d";
    }

    return {
        fillColor: color,
        color: "#111",
        weight: 1,
        fillOpacity: 0.65
    };
}
""")

# ============================================================
# 6. FIGURAS PLOTLY
# ============================================================

df_agua_plot = df_muni.sort_values("altura_media_agua", ascending=False).head(20)
fig_agua = px.bar(
    df_agua_plot,
    x="nombre_norm",
    y="altura_media_agua",
    title="Altura media de agua",
    labels={"nombre_norm": "Municipio", "altura_media_agua": "Altura media del agua (m)"}
).update_layout(
    template="plotly_white",
    xaxis_tickangle=-45,
    margin=dict(l=40, r=20, t=60, b=120)
)

df_agua_max_plot = df_muni.sort_values("altura_max_agua", ascending=False).head(20)
fig_agua_max = px.bar(
    df_agua_max_plot,
    x="nombre_norm",
    y="altura_max_agua",
    title="Altura máxima de agua (Top 20)",
    labels={"nombre_norm": "Municipio", "altura_max_agua": "Altura máxima del agua (m)"}
).update_layout(
    template="plotly_white",
    xaxis_tickangle=-45,
    margin=dict(l=40, r=20, t=60, b=120)
)

# ============================================================
# 7. MÉTRICAS PARA RANKINGS Y COMPARATIVAS
# ============================================================

METRICAS_RANK = {
    "danio_medio": "Daño medio",
    "danio_max": "Daño máximo",
    "igd_medio": "IGD medio",
    "igd_max": "IGD máximo",
    "altura_media_agua": "Altura media de agua",
    "altura_max_agua": "Altura máxima de agua",
    "edificios": "Nº de edificios inspeccionados",
}

LABELS_EJES = {
    "nombre_norm": "Municipio",
    "valor": "Valor",
    "igd_medio": "IGD medio",
    "igd_max": "IGD máximo",
    "danio_medio": "Daño medio",
    "danio_max": "Daño máximo",
    "altura_media_agua": "Altura media del agua (m)",
    "altura_max_agua": "Altura máxima del agua (m)",
    "edificios": "Nº de edificios inspeccionados",
    "variable_legible": "Métrica",
}

METRICAS_COMPARA = ["igd_medio", "danio_medio", "altura_media_agua"]

# ============================================================
# 8. ASISTENTE IA: NL → SQL (OPCIONAL)
#    - En Vercel: déjalo desactivado (ENABLE_ASSISTANT=0)
# ============================================================

if ENABLE_ASSISTANT:
    import dspy
    import litellm
    from dspy.teleprompt import BootstrapFewShot

    # PARCHE LM STUDIO (evita response_format)
    original_completion = litellm.completion

    def patched_completion(*args, **kwargs):
        kwargs.pop("response_format", None)
        return original_completion(*args, **kwargs)

    litellm.completion = patched_completion

    MODEL_NAME = os.getenv("LM_MODEL_NAME", "openai/phi1.5-quantized-llm")
    LM_API_BASE = os.getenv("LM_API_BASE", "http://localhost:1234/v1")
    LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")

    lm = dspy.LM(
        MODEL_NAME,
        api_base=LM_API_BASE,
        api_key=LM_API_KEY,
        model_type="chat",
        temperature=0.2,
        max_tokens=600,
    )

    # por seguridad adicional
    lm.kwargs.pop("response_format", None)
    dspy.configure(lm=lm)

    print("DSPy configurado para NL → SQL")

    # HELPERS DUCKDB: esquema seguro
    def connect_duckdb(path: str):
        return duckdb.connect(path, read_only=True)

    def inspect_schema(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        tables = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='main'
            """
        ).fetchall()

        rows = []
        for (tname,) in tables:
            cols = con.execute(
                f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='main' AND table_name='{tname}'
                """
            ).fetchall()
            for col_name, data_type in cols:
                rows.append((tname, col_name, data_type))
        return pd.DataFrame(rows, columns=["table_name", "column_name", "data_type"])

    def format_schema_for_llm(schema_df: pd.DataFrame) -> str:
        if schema_df.empty:
            return "NO_TABLES"
        parts = []
        for tname, group in schema_df.groupby("table_name"):
            cols = ", ".join(group["column_name"].tolist())
            parts.append(f"table {tname}: {cols}")
        return "; ".join(parts)

    def schema_para_prompt(prompt: str) -> str:
        p = (prompt or "").lower()
        if "municipio" in p:
            return (
                "table datos_municipio: "
                "nombre_norm, danio_medio, danio_max, "
                "igd_medio, igd_max, edificios, "
                "altura_media_agua, altura_max_agua"
            )
        return (
            "table datos_procesados: "
            "lat, lon, danos_total, cotaagua_normalizada"
        )

    METRIC_MAP = {
        "daño medio": "danio_medio",
        "daño máximo": "danio_max",
        "igd medio": "igd_medio",
        "igd máximo": "igd_max",
        "altura media": "altura_media_agua",
        "altura máxima": "altura_max_agua",
        "edificios": "edificios",
    }

    def detectar_metrica(prompt: str) -> str | None:
        p = (prompt or "").lower()
        for texto, col in METRIC_MAP.items():
            if texto in p:
                return col
        return None

    def plantilla_sql(prompt: str) -> str | None:
        p = (prompt or "").lower()
        if any(x in p for x in ["mayor", "máximo", "más alto"]):
            for texto, col in METRIC_MAP.items():
                if texto in p:
                    return (
                        f"SELECT nombre_norm, {col} "
                        f"FROM datos_municipio "
                        f"ORDER BY {col} DESC "
                        f"LIMIT 1"
                    )
        if "por municipio" in p or "cada municipio" in p or "todos los municipios" in p:
            for texto, col in METRIC_MAP.items():
                if texto in p:
                    return (
                        f"SELECT nombre_norm, {col} "
                        f"FROM datos_municipio "
                        f"ORDER BY nombre_norm"
                    )
        return None

    def sql_fallback(prompt: str) -> str | None:
        p = (prompt or "").lower()
        if "nombre" in p and "municipio" in p:
            return "SELECT nombre_norm FROM datos_municipio"
        if "mayor" in p and "daño medio" in p:
            return (
                "SELECT nombre_norm, danio_medio "
                "FROM datos_municipio "
                "ORDER BY danio_medio DESC "
                "LIMIT 1"
            )
        return None

    def sql_few_shot_trainset():
        examples = [
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm",
                request="Dime el nombre de los municipios afectados",
                sql_query="SELECT nombre_norm FROM datos_municipio"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, edificios",
                request="Número de edificios inspeccionados por municipio",
                sql_query="SELECT nombre_norm, edificios FROM datos_municipio ORDER BY nombre_norm"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, altura_media_agua",
                request="Altura media de agua por municipio",
                sql_query="SELECT nombre_norm, altura_media_agua FROM datos_municipio ORDER BY nombre_norm"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, danio_medio",
                request="Daño medio por municipio",
                sql_query="SELECT nombre_norm, danio_medio FROM datos_municipio ORDER BY nombre_norm"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, danio_medio",
                request="¿Cuál es el municipio con mayor daño medio?",
                sql_query="SELECT nombre_norm, danio_medio FROM datos_municipio ORDER BY danio_medio DESC LIMIT 1"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, altura_max_agua",
                request="Municipio con mayor altura máxima de agua",
                sql_query="SELECT nombre_norm, altura_max_agua FROM datos_municipio ORDER BY altura_max_agua DESC LIMIT 1"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, altura_media_agua",
                request="Dame los 5 municipios con mayor altura media de agua",
                sql_query="SELECT nombre_norm, altura_media_agua FROM datos_municipio ORDER BY altura_media_agua DESC LIMIT 5"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, edificios",
                request="Los 3 municipios con más edificios inspeccionados",
                sql_query="SELECT nombre_norm, edificios FROM datos_municipio ORDER BY edificios DESC LIMIT 3"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, edificios",
                request="Municipios con más de 100 edificios inspeccionados",
                sql_query="SELECT nombre_norm, edificios FROM datos_municipio WHERE edificios > 100 ORDER BY edificios DESC"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, danio_max",
                request="Ordena los municipios por daño máximo",
                sql_query="SELECT nombre_norm, danio_max FROM datos_municipio ORDER BY danio_max DESC"
            ),
            dspy.Example(
                df_schema="table datos_municipio: nombre_norm, igd_medio",
                request="Ordena los municipios por IGD medio",
                sql_query="SELECT nombre_norm, igd_medio FROM datos_municipio ORDER BY igd_medio DESC"
            ),
        ]
        return [ex.with_inputs("df_schema", "request") for ex in examples]

    def metric_sql(gold, pred, trace=None):
        if not isinstance(gold, str) or not isinstance(pred, str):
            return 0.0
        score = 0.0
        p = pred.lower()
        if "select" in p:
            score += 0.4
        if "from" in p:
            score += 0.4
        for token in gold.lower().replace("(", " ").replace(")", " ").split():
            if len(token) > 4 and token in p:
                score += 0.2
                break
        return min(score, 1.0)

    class SQLSignature(dspy.Signature):
        df_schema: str = dspy.InputField(desc="Database schema")
        request: str = dspy.InputField(desc="User question in natural language")
        sql_query: str = dspy.OutputField(format="text")

    class SQLModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(SQLSignature)

        def forward(self, df_schema: str, request: str) -> str:
            out = self.predict(
                df_schema=df_schema,
                request=(
                    "Generate ONE valid DuckDB SQL query.\n"
                    "STRICT RULES:\n"
                    "- Output ONLY SQL\n"
                    "- DO NOT output JSON\n"
                    "- DO NOT output dictionaries or lists\n"
                    "- DO NOT describe columns\n"
                    "- DO NOT explain anything\n"
                    "- MUST start with SELECT\n"
                    "- MUST contain FROM\n"
                    "- Use ONLY tables and columns from the schema\n\n"
                    f"Schema:\n{df_schema}\n\n"
                    f"Question:\n{request}"
                )
            )

            sql = (out.sql_query or "").strip()
            if sql.startswith("```"):
                sql = sql.strip("`")
                if sql.lower().startswith("sql"):
                    sql = sql.split("\n", 1)[-1]
            return sql.rstrip(";").strip()

    base_sql_module = SQLModule()
    teleprompter = BootstrapFewShot(metric=metric_sql, max_bootstrapped_demos=1, max_labeled_demos=2)

    SQL_PROGRAM = teleprompter.compile(base_sql_module, trainset=sql_few_shot_trainset())
    print("DSPy SQL_PROGRAM compilado correctamente")

    def normalizar_sql(sql: str, prompt: str) -> str:
        p = (prompt or "").lower()
        sql = (sql or "").strip()

        if any(x in p for x in ["por municipio", "cada municipio", "todos los municipios"]):
            sql = re.sub(r"limit\s+\d+", "", sql, flags=re.IGNORECASE)

        for texto, col in METRIC_MAP.items():
            if texto in p:
                sql = re.sub(r"order by\s+\w+", f"ORDER BY {col}", sql, flags=re.IGNORECASE)

        m = re.search(r"(los|las)\s+(\d+)", p)
        if m:
            n = m.group(2)
            if "limit" not in sql.lower():
                sql = f"{sql.rstrip()} LIMIT {n}"

        m = re.search(r"más de (\d+)", p)
        if m and "edificio" in p:
            n = m.group(1)
            sql = (
                "SELECT nombre_norm, edificios "
                "FROM datos_municipio "
                f"WHERE edificios > {n} "
                "ORDER BY edificios DESC"
            )

        if "limit" in sql.lower() and not any(
            x in p for x in ["top", "los", "las", "mayor", "menor", "máximo", "mínimo"]
        ):
            sql = re.sub(r"limit\s+\d+", "", sql, flags=re.IGNORECASE)

        return sql.strip()

    def validar_sql(sql: str, prompt: str) -> str | None:
        if not isinstance(sql, str):
            return None

        s = sql.lower().strip()
        p = (prompt or "").lower()

        if not s.startswith("select"):
            return None
        if " from " not in s:
            return None

        if any(x in s for x in ["insert", "update", "delete", "drop", "alter"]):
            return None

        if "{" in s or "}" in s:
            return None

        if "municipio" in p and "datos_municipio" not in s:
            return None

        if "where" in s and not any(w in p for w in ["donde", "con", "que tenga", "filtra", "más de", "menos de"]):
            return None

        return sql

    def generar_sql_desde_prompt(prompt: str) -> str | None:
        if not prompt:
            return None

        schema_llm = schema_para_prompt(prompt)

        metrica = detectar_metrica(prompt)
        if metrica:
            schema_llm = f"table datos_municipio: nombre_norm, {metrica}"

        sql_query = plantilla_sql(prompt)
        if sql_query is None:
            sql_query = sql_fallback(prompt)

        if sql_query is None:
            sql_query = SQL_PROGRAM(df_schema=schema_llm, request=prompt)

        sql_query = normalizar_sql(sql_query, prompt)
        return validar_sql(sql_query, prompt)

# ============================================================
# 9. DASH APP — LAYOUT
# ============================================================

tabs_children = [
    # ====================================================
    # TAB 1 — MAPA
    # ====================================================
    dcc.Tab(
        label="Mapa por municipio",
        value="tab-mapa",
        children=[
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Variable a representar", className="fw-semibold"),
                                        dcc.Dropdown(
                                            id="selector",
                                            options=[
                                                {"label": "IGD medio", "value": "igd_medio"},
                                                {"label": "IGD máximo", "value": "igd_max"},
                                                {"label": "Daño medio", "value": "danio_medio"},
                                                {"label": "Daño máximo", "value": "danio_max"},
                                                {"label": "Edificios inspeccionados", "value": "edificios"},
                                                {"label": "Altura media agua", "value": "altura_media_agua"},
                                                {"label": "Altura máxima agua", "value": "altura_max_agua"},
                                            ],
                                            value="igd_medio",
                                            clearable=False,
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Alert(
                                            [
                                                html.I(className="bi bi-geo-alt me-2"),
                                                "Capa municipal (coropletas) + puntos de edificios. Haz zoom y pulsa sobre un municipio para ver métricas.",
                                            ],
                                            color="light",
                                            className="mb-0",
                                        )
                                    ],
                                    md=8,
                                ),
                            ],
                            className="mb-3",
                            align="center",
                        ),
                        html.Div(
                            [
                                dl.Map(
                                    id="mapa",
                                    center=[39.5, -0.4],
                                    zoom=8,
                                    style={"width": "100%", "height": "650px", "borderRadius": "10px"},
                                    children=[
                                        dl.TileLayer(
                                            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                                        ),
                                        dl.Pane(id="pane-municipios", name="municipios", style={"zIndex": 200}),
                                        dl.Pane(id="pane-puntos", name="puntos", style={"zIndex": 500}),
                                        dl.GeoJSON(
                                            id="geojson",
                                            data=geojson_filtrado,
                                            pane="municipios",
                                            zoomToBounds=True,
                                            zoomToBoundsOnClick=True,
                                            hoverStyle={"weight": 3, "color": "black", "fillOpacity": 0.75},
                                            style=geojson_style,
                                            hideout={"variable": "igd_medio"},
                                        ),
                                        dl.LayerGroup(
                                            id="layer-puntos",
                                            pane="puntos",
                                            children=markers,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="leyenda",
                                    className="shadow-sm",
                                    style={
                                        "position": "absolute",
                                        "bottom": "20px",
                                        "left": "20px",
                                        "background": "white",
                                        "padding": "12px",
                                        "borderRadius": "10px",
                                        "fontSize": "13px",
                                        "zIndex": "500",
                                        "minWidth": "190px",
                                    },
                                ),
                            ],
                            style={"position": "relative"},
                        ),
                    ]
                ),
                className=CARD_CLASS,
            )
        ],
    ),

    # ====================================================
    # TAB 2 — RANKINGS Y COMPARATIVAS
    # ====================================================
    dcc.Tab(
        label="Rankings y comparativas",
        value="tab-ranking",
        children=[
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Rankings y comparativas entre municipios", className="fw-bold mb-1"),
                        html.P("Ordena municipios por métricas y compara varios a la vez.", className="text-muted"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Opciones", className="fw-bold"),
                                                    html.Label("Métrica de ranking"),
                                                    dcc.Dropdown(
                                                        id="rank-metrica",
                                                        options=[
                                                            {"label": label, "value": value}
                                                            for value, label in METRICAS_RANK.items()
                                                        ],
                                                        value="danio_medio",
                                                        clearable=False,
                                                    ),
                                                    html.Hr(),
                                                    html.Label("Municipios a comparar"),
                                                    dcc.Dropdown(
                                                        id="rank-munis",
                                                        options=[
                                                            {"label": m, "value": m}
                                                            for m in sorted(df_muni["nombre_norm"].unique())
                                                        ],
                                                        multi=True,
                                                        value=sorted(df_muni["nombre_norm"].unique())[:3],
                                                    ),
                                                    html.Small(
                                                        "Sugerencia: 3–6 municipios para buena legibilidad.",
                                                        className="text-muted",
                                                    ),
                                                ]
                                            ),
                                            className="shadow-sm",
                                        )
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Graph(id="rank-bar", style={"height": "420px"})
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dash_table.DataTable(
                                                                id="rank-table",
                                                                page_size=10,
                                                                style_table={"overflowX": "auto"},
                                                                style_header={
                                                                    "fontWeight": "600",
                                                                    "backgroundColor": "#f8f9fa",
                                                                },
                                                                style_cell={"padding": "6px", "fontSize": 12},
                                                            )
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Graph(id="rank-compare", style={"height": "420px"})
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Graph(id="rank-radar", style={"height": "420px"})
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Graph(id="rank-treemap", style={"height": "420px"})
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                    width=12,
                                                )
                                            ]
                                        ),
                                    ],
                                    md=8,
                                ),
                            ]
                        ),
                    ]
                ),
                className=CARD_CLASS,
            )
        ],
    ),
]

# TAB 3 — solo si el asistente está activado
if ENABLE_ASSISTANT:
    tabs_children.append(
        dcc.Tab(
            label="Asistente IA",
            value="tab-asistente",
            children=[
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Asistente IA para consultas (NL → SQL)", className="fw-bold mb-1"),
                            html.P(
                                "Escribe una pregunta en lenguaje natural y el sistema genera SQL seguro para DuckDB.",
                                className="text-muted",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Pregunta (lenguaje natural)", className="fw-semibold"),
                                            dcc.Textarea(
                                                id="asistente-prompt",
                                                style={"width": "100%", "height": "90px"},
                                                placeholder="Ej.: ¿Cuál es el municipio con mayor daño medio?",
                                            ),
                                            dbc.Button(
                                                [html.I(className="bi bi-play-fill me-1"), "Generar SQL y ejecutar"],
                                                id="asistente-run",
                                                color="primary",
                                                className="mt-2",
                                                n_clicks=0,
                                            ),
                                            html.Div(id="asistente-error", className="text-danger mt-2"),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("SQL generado", className="fw-semibold"),
                                            dcc.Textarea(
                                                id="asistente-sql",
                                                style={
                                                    "width": "100%",
                                                    "height": "140px",
                                                    "fontFamily": "monospace",
                                                    "backgroundColor": "#f8f9fa",
                                                },
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="g-3 mb-3",
                            ),
                            html.H6("Resultado de la consulta", className="fw-bold"),
                            dash_table.DataTable(
                                id="asistente-tabla",
                                page_size=12,
                                style_table={"overflowX": "auto"},
                                style_header={"fontWeight": "600", "backgroundColor": "#f8f9fa"},
                                style_cell={"padding": "6px", "fontSize": 12},
                            ),
                            dcc.Store(id="asistente-schema"),
                        ]
                    ),
                    className=CARD_CLASS,
                )
            ],
        )
    )

app.layout = dbc.Container(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H2(
                                        "Dashboard DANA — Edificios afectados en la Comunitat Valenciana",
                                        className="fw-bold mb-1",
                                    ),
                                    html.P(
                                        "Incluye puntos geolocalizados, agregación por municipio, rankings y comparativas.",
                                        className="text-muted mb-2",
                                    ),
                                    dbc.Badge("Datos municipales", color="primary", className="me-2"),
                                    dbc.Badge("Mapa interactivo", color="info", className="me-2"),
                                    # dbc.Badge("Asistente IA", color="success"),
                                ],
                                md=9,
                            ),
                            dbc.Col(
                                [
                                    dbc.Alert(
                                        [
                                            html.I(className="bi bi-info-circle me-2"),
                                            "Consejo: usa el selector del mapa para cambiar la variable y consulta detalles al hacer click en municipios o puntos.",
                                        ],
                                        color="light",
                                        className="mb-0",
                                    )
                                ],
                                md=3,
                            ),
                        ],
                        align="center",
                    )
                ]
            ),
            className=f"{CARD_CLASS} hero-dana",
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-mapa",
            children=tabs_children,
        ),
    ],
    fluid=True,
    style=PAGE_STYLE,
)

# ============================================================
# 11. CALLBACKS
# ============================================================

@app.callback(
    Output("geojson", "hideout"),
    Input("selector", "value")
)
def actualizar_mapa(variable):
    return {"variable": variable}


@app.callback(
    Output("leyenda", "children"),
    Input("selector", "value")
)
def actualizar_leyenda(variable):

    if variable in ("igd_medio", "igd_max"):
        titulo = "IGD"
        rangos = [
            ("IGD < 1", "#d4eeff"),
            ("1–4", "#74c0e3"),
            ("4–10", "#1f78b4"),
            ("≥10", "#08306b")
        ]

    elif variable == "danio_medio":
        titulo = "Daño medio"
        rangos = [
            ("< 0.5", "#fee5d9"),
            ("0.5–1.0", "#fcae91"),
            ("1–2", "#fb6a4a"),
            ("> 2", "#cb181d")
        ]

    elif variable == "danio_max":
        titulo = "Daño máximo"
        rangos = [
            ("< 2", "#fee5d9"),
            ("2–5", "#fcae91"),
            ("5–10", "#fb6a4a"),
            ("> 10", "#cb181d")
        ]

    elif variable == "edificios":
        titulo = "Nº edificios"
        rangos = [
            ("< 50", "#fee5d9"),
            ("50–150", "#fcae91"),
            ("150–400", "#fb6a4a"),
            ("> 400", "#cb181d")
        ]

    else:
        titulo = "Altura agua"
        rangos = [
            ("< 0.2", "#fee5d9"),
            ("0.2–0.6", "#fcae91"),
            ("0.6–1.2", "#fb6a4a"),
            ("> 1.2", "#cb181d")
        ]

    bloques = [
        html.Div(
            [
                html.Div(
                    style={
                        "width": "18px",
                        "height": "18px",
                        "background": color,
                        "display": "inline-block",
                        "marginRight": "10px",
                        "border": "1px solid #111",
                        "borderRadius": "4px",
                        "verticalAlign": "middle",
                    }
                ),
                html.Span(label, style={"verticalAlign": "middle"})
            ],
            style={"marginBottom": "6px"}
        )
        for label, color in rangos
    ]

    return [html.Div(html.B(titulo), style={"marginBottom": "6px"}), *bloques]


# ---------- Asistente IA callbacks (solo si ENABLE_ASSISTANT) ----------

if ENABLE_ASSISTANT:

    @app.callback(
        Output("asistente-schema", "data"),
        Input("tabs", "value"),
    )
    def cargar_schema(tab):
        if tab != "tab-asistente":
            return no_update
        try:
            con_local = connect_duckdb(DB_PATH)
            schema_df = inspect_schema(con_local)
            con_local.close()
            schema_str = format_schema_for_llm(schema_df)
            return schema_str
        except Exception as e:
            return f"ERROR_SCHEMA: {e}"

    @app.callback(
        Output("asistente-sql", "value"),
        Output("asistente-tabla", "columns"),
        Output("asistente-tabla", "data"),
        Output("asistente-error", "children"),
        Input("asistente-run", "n_clicks"),
        State("asistente-prompt", "value"),
        State("asistente-schema", "data"),
        prevent_initial_call=True
    )
    def ejecutar_asistente(n, prompt, schema_str):

        if not prompt:
            return "", [], [], "Escribe una consulta en lenguaje natural."

        if not schema_str or (isinstance(schema_str, str) and schema_str.startswith("ERROR")):
            return "", [], [], "No se pudo cargar el esquema."

        try:
            sql_query = generar_sql_desde_prompt(prompt)
            if not sql_query:
                return "", [], [], "No he podido generar una consulta SQL fiable."
        except Exception as e:
            return "", [], [], f"Error generando SQL: {e}"

        try:
            con_local = duckdb.connect(DB_PATH, read_only=True)
            df_res = con_local.execute(sql_query).df()
            con_local.close()
        except Exception as e:
            return sql_query, [], [], f"Error ejecutando SQL: {e}"

        if df_res.empty:
            return sql_query, [], [], "Consulta correcta, pero sin filas."

        columns = [{"name": c, "id": c} for c in df_res.columns]
        data = df_res.to_dict("records")

        return sql_query, columns, data, ""


# ---------------- RANKINGS Y COMPARATIVAS ----------------

@app.callback(
    Output("rank-bar", "figure"),
    Output("rank-table", "columns"),
    Output("rank-table", "data"),
    Output("rank-compare", "figure"),
    Output("rank-radar", "figure"),
    Output("rank-treemap", "figure"),
    Input("rank-metrica", "value"),
    Input("rank-munis", "value"),
)
def actualizar_rankings(metrica, munis_sel):

    if metrica not in METRICAS_RANK:
        invert = {v: k for k, v in METRICAS_RANK.items()}
        metrica = invert.get(metrica, "danio_medio")

    df_top = df_muni.sort_values(metrica, ascending=False)

    fig_rank = px.bar(
        df_top,
        x="nombre_norm",
        y=metrica,
        title=f"Ranking de municipios por {METRICAS_RANK[metrica]}",
        labels={"nombre_norm": "Municipio", metrica: METRICAS_RANK[metrica]},
    ).update_layout(
        template="plotly_white",
        height=420,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=20, t=60, b=120)
    )

    if metrica == "edificios":
        df_tabla = df_top[["nombre_norm", "edificios"]].copy()
        df_tabla.rename(columns={"nombre_norm": "Municipio", "edificios": "Nº edificios"}, inplace=True)
    else:
        df_tabla = df_top[["nombre_norm", metrica, "edificios"]].copy()
        df_tabla.rename(
            columns={
                "nombre_norm": "Municipio",
                metrica: METRICAS_RANK[metrica],
                "edificios": "Nº edificios",
            },
            inplace=True
        )

    columns = [{"name": c, "id": c} for c in df_tabla.columns]
    data = df_tabla.to_dict("records")

    if not munis_sel:
        munis_sel = df_top["nombre_norm"].tolist()

    df_sel = df_muni[df_muni["nombre_norm"].isin(munis_sel)]

    df_comp = df_sel[["nombre_norm"] + METRICAS_COMPARA].melt(
        id_vars="nombre_norm",
        var_name="variable",
        value_name="valor",
    )

    nombre_var = {
        "igd_medio": "IGD medio",
        "danio_medio": "Daño medio",
        "altura_media_agua": "Altura media agua",
    }
    df_comp["variable_legible"] = df_comp["variable"].map(nombre_var)

    fig_comp = px.bar(
        df_comp,
        x="nombre_norm",
        y="valor",
        color="variable_legible",
        barmode="group",
        title="Comparativa de métricas entre municipios seleccionados",
        labels={
            "nombre_norm": "Municipio",
            "valor": "Valor de la métrica",
            "variable_legible": "Métrica",
        },
    ).update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=120),
        xaxis_tickangle=-45,
    )

    fig_radar = px.line_polar(
        df_comp,
        r="valor",
        theta="variable_legible",
        color="nombre_norm",
        line_close=True,
        title="Radar de métricas por municipio",
    )
    fig_radar.update_traces(fill="toself", opacity=0.6)
    fig_radar.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(tickfont=dict(size=11), showline=True, linewidth=1, gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        legend_title_text="Municipio",
    )

    fig_treemap = px.treemap(
        df_muni,
        path=["nombre_norm"],
        values=metrica,
        color=metrica,
        title=f"Treemap por {METRICAS_RANK[metrica]}",
        labels={"nombre_norm": "Municipio", metrica: METRICAS_RANK[metrica]},
    )

    fig_treemap.update_traces(
        textinfo="label",
        root_color="rgba(0,0,0,0)",
        textfont=dict(size=14, color="white", family="Arial"),
        marker=dict(line=dict(width=2, color="white")),
        hovertemplate=(
            "<b>%{label}</b><br>"
            + METRICAS_RANK[metrica]
            + ": %{value:.2f}<extra></extra>"
        )
    )
    fig_treemap.update_layout(
        template="plotly_white",
        margin=dict(t=60, l=10, r=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        coloraxis_colorbar=dict(
            title=dict(text=METRICAS_RANK[metrica], font=dict(size=12)),
            ticks="outside",
            tickfont=dict(size=11),
        ),
    )

    return fig_rank, columns, data, fig_comp, fig_radar, fig_treemap


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    host = "0.0.0.0"

    print("\nDashboard DANA en ejecución")
    print(f"Escuchando en {host}:{port}\n")

    app.run(
        debug=False,
        host=host,
        port=port
    )