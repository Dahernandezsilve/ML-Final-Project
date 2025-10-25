import logging
import pandas as pd
from datasets import load_dataset

log = logging.getLogger(__name__)

def load_raw() -> pd.DataFrame:
    """
    Carga el dataset HF: biglam/spanish_golden_age_sonnets
    """
    log.info("Cargando dataset HF: biglam/spanish_golden_age_sonnets ...")
    ds = load_dataset("biglam/spanish_golden_age_sonnets")
    df = pd.DataFrame(ds["train"]) # type: ignore
    expected = ["author", "sonnet_text"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing}. Columnas reales: {list(df.columns)}")
    log.info("Dataset cargado. Filas=%d", len(df))
    return df

def overview(df: pd.DataFrame) -> dict:
    info = {
        "autores_unicos": int(df["author"].nunique()),
        "top20_autores": df["author"].value_counts().head(20).to_dict(),
        "columnas": list(df.columns),
    }
    log.debug("overview() -> %s", info)
    return info
