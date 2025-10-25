import logging, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

def basic_clean(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def prepare(df: pd.DataFrame, top_k: int = 8, seed: int = 42):
    """
    - Limpieza básica de texto (lower + compresión de espacios)
    - Balanceo por mapeo: TOP-K autores; resto -> 'other'
    - Split estratificado 80/10/10 en el label mapeado
    """
    df = df.copy()
    df["text_clean"] = df["sonnet_text"].astype(str).apply(basic_clean)

    counts = df["author"].value_counts()
    top_authors = set(counts.head(top_k).index)
    df["author_mapped"] = df["author"].apply(lambda a: a if a in top_authors else "other")

    X = df["text_clean"].values
    y = df["author_mapped"].values

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed) # type: ignore
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed
    )

    labels = sorted(list(pd.Series(y).unique()))
    log.info("Autores mapeados: %s", {k:int(v) for k,v in pd.Series(y).value_counts().to_dict().items()})
    log.debug("Split -> train=%d, val=%d, test=%d", len(Xtr), len(Xva), len(Xte))
    return (Xtr, ytr), (Xva, yva), (Xte, yte), labels
