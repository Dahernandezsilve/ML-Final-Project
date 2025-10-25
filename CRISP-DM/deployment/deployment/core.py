import logging
import os
import joblib

log = logging.getLogger(__name__)

def save_model(model, path: str = "artifacts/model_discriminador.joblib") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log.info("Modelo (discriminador) persistido en: %s", path)
    return path

def load_model(path: str = "artifacts/model_discriminador.joblib"):
    log.info("Cargando modelo (discriminador) desde: %s", path)
    return joblib.load(path)

def predict(model, texts):
    log.debug("Inferencia sobre %d textos...", len(texts))
    return model.predict(texts)

def save_ngram_model(model_tuple, path: str = "artifacts/model_ngram.joblib") -> str:
    """
    Guarda el modelo n-gramas como un tuple: (starts, counts, n)
      - starts: list[tuple[str, ...]]
      - counts: defaultdict(Counter)
      - n: int
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_tuple, path)
    log.info("Modelo generador n-gramas persistido en: %s", path)
    return path

def load_ngram_model(path: str = "artifacts/model_ngram.joblib"):
    """
    Carga el tuple (starts, counts, n) del modelo n-gramas.
    """
    log.info("Cargando modelo n-gramas desde: %s", path)
    return joblib.load(path)
