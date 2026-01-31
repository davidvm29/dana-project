# precompute_points.py
# ------------------------------------------------------------
# Genera data/puntos_filtrados.parquet a partir de:
# - DuckDB: data/db/dana.duckdb (tabla datos_procesados)
# - GeoJSON: cv_municipios.geojson
#
# Replica tu lógica:
# - Filtra puntos con lat/lon/nombre_norm
# - Carga geometrías municipales
# - Reproyecta a CRS métrico (EPSG:25830)
# - Calcula distancia al borde municipal
# - Mantiene puntos a <= DIST_MAX_METROS
# - Guarda Parquet final con columnas necesarias para el mapa
# ------------------------------------------------------------

import os
import duckdb
import pandas as pd
import geopandas as gpd

DB_PATH = os.getenv("DANA_DB_PATH", "data/db/dana.duckdb")
GEOJSON_PATH = os.getenv("DANA_GEOJSON_PATH", "cv_municipios.geojson")
OUT_PATH = os.getenv("DANA_PUNTOS_OUT", "data/puntos_filtrados.parquet")

DIST_MAX_METROS = int(os.getenv("DIST_MAX_METROS", "3000"))  # 3km por defecto

# Columnas mínimas que el dashboard espera en df_puntos
REQUIRED_OUT_COLS = [
    "lat",
    "lon",
    "nombremunicipio",
    "igd",
    "danos_total",
    "cotaagua_normalizada",
    "imagencatastro",
]


def main():
    print("== Precompute puntos filtrados ==")
    print(f"DB: {DB_PATH}")
    print(f"GeoJSON: {GEOJSON_PATH}")
    print(f"Salida: {OUT_PATH}")
    print(f"DIST_MAX_METROS: {DIST_MAX_METROS}")

    # 1) Cargar tabla datos_procesados
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("SELECT * FROM datos_procesados").fetchdf()
    con.close()
    print(f"Filas datos_procesados: {len(df):,}")

    # 2) Filtrar puntos con lat/lon/nombre_norm
    df_geo = df.dropna(subset=["lat", "lon", "nombre_norm"]).copy()
    print(f"Filas con lat/lon/nombre_norm: {len(df_geo):,}")

    # 3) GeoDataFrame de puntos (WGS84)
    gdf_puntos = gpd.GeoDataFrame(
        df_geo,
        geometry=gpd.points_from_xy(df_geo["lon"], df_geo["lat"]),
        crs="EPSG:4326"
    )

    # 4) Cargar municipios y normalizar columna
    gdf_muni = gpd.read_file(GEOJSON_PATH)
    # en tu geojson el nombre está en "municipio"
    if "municipio" in gdf_muni.columns and "nombre_norm" not in gdf_muni.columns:
        gdf_muni = gdf_muni.rename(columns={"municipio": "nombre_norm"})

    if "nombre_norm" not in gdf_muni.columns:
        raise ValueError(
            f"El GeoJSON no tiene 'municipio' ni 'nombre_norm'. Columnas: {list(gdf_muni.columns)}"
        )

    gdf_muni = gdf_muni[["nombre_norm", "geometry"]].copy()

    # 5) Reproyección a CRS métrico
    gdf_puntos_m = gdf_puntos.to_crs("EPSG:25830")  # UTM 30N
    gdf_muni_m = gdf_muni.to_crs("EPSG:25830")

    # 6) Unificar geometrías por municipio (dissolve)
    gdf_muni_m = gdf_muni_m.dissolve(by="nombre_norm", as_index=False)

    # 7) Asignar geometría municipal a cada punto por nombre_norm
    muni_geom = gdf_muni_m.set_index("nombre_norm")["geometry"]
    gdf_puntos_m["geom_muni"] = gdf_puntos_m["nombre_norm"].map(muni_geom)
    gdf_puntos_m = gdf_puntos_m.dropna(subset=["geom_muni"]).copy()
    print(f"Filas con municipio mapeado: {len(gdf_puntos_m):,}")

    # 8) Distancia al borde municipal
    # Nota: distance devuelve 0 si el punto está dentro.
    gdf_puntos_m["dist_borde_m"] = gdf_puntos_m.geometry.distance(gdf_puntos_m["geom_muni"])

    # 9) Filtrar por distancia máxima
    gdf_filtrado = gdf_puntos_m[gdf_puntos_m["dist_borde_m"] <= DIST_MAX_METROS].copy()
    print(f"Filas tras filtro distancia <= {DIST_MAX_METROS}m: {len(gdf_filtrado):,}")

    # 10) Convertir a DataFrame y quitar geometrías extra
    df_out = pd.DataFrame(gdf_filtrado.drop(columns=["geometry", "geom_muni"], errors="ignore"))

    # 11) Asegurar columnas mínimas
    missing = [c for c in REQUIRED_OUT_COLS if c not in df_out.columns]
    if missing:
        raise ValueError(
            "Faltan columnas necesarias para el dashboard en datos_procesados: "
            f"{missing}\n"
            "Solución: ajusta REQUIRED_OUT_COLS o revisa nombres de columnas."
        )

    # 12) Guardar parquet
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_parquet(OUT_PATH, index=False)
    print(f"OK guardado: {OUT_PATH} (filas: {len(df_out):,}, cols: {len(df_out.columns)})")


if __name__ == "__main__":
    main()
