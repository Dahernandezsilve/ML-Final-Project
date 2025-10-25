import logging
log = logging.getLogger(__name__)

def goals():
    """
    Objetivo técnico (antagonistas):
      - Generador: producir textos poéticos originales (estilo/métrica/vocabulario de autores clásicos).
      - Discriminador: identificar si un poema es auténtico o generado por el modelo.
    """
    info = {
        "objetivo_tecnico": (
            "Sistema de PLN con dos modelos antagonistas: "
            "un generador (n-gramas) y un discriminador (TF-IDF+LinearSVC). "
            "Se busca explorar patrones estilísticos del Siglo de Oro y su generación/identificación."
        ),
        "kpi": "F1_macro en test para el discriminador; BLEU/ROUGE para el generador",
        "criterio_exito": "Discriminador con F1_macro competitivo y métricas de similaridad > baseline",
        "riesgos": ["desbalance de autores", "fugas de información", "limitaciones del baseline"],
    }
    log.debug("goals() -> %s", info)
    return info
