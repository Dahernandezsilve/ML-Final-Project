import logging
from business_understanding.business_understanding.core import goals
from data_understanding.data_understanding.core import load_raw, overview
from data_preparation.data_preparation.core import prepare
from modeling.modeling.core import train_discriminator, build_ngram_model, generate_text
from evaluation.evaluation.core import evaluate_classifier, evaluate_generator
from deployment.deployment.core import save_model, save_ngram_model, load_ngram_model

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def ensure_nltk():
    import nltk
    # (path_en_nltk, recurso_para_descargar)
    resources = [
        ("corpora/stopwords",   "stopwords"),
        ("tokenizers/punkt",    "punkt"),
        ("tokenizers/punkt_tab","punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"🔧 Descargando NLTK: {name} …")
            nltk.download(name)

    # Sanity check (forza carga sin fallar silenciosamente)
    from nltk.corpus import stopwords as _sw
    from nltk.tokenize import word_tokenize as _wt
    _ = _sw.words("spanish")
    _ = _wt("hola mundo.")


def main():
    setup_logging()
    log = logging.getLogger("pipeline")

    log.info("== Fase 1: Comprensión del negocio ==")
    g = goals()
    log.info("Objetivo técnico: %s", g["objetivo_tecnico"])

    log.info("== Fase 2: Comprensión de los datos ==")
    df = load_raw()  # dataset HF
    log.info("Dataset cargado: %d filas, columnas=%s", len(df), list(df.columns))
    log.info("Overview: %s", overview(df))

    log.info("== descargando/cargando recursos NLTK (si faltan) ==")
    ensure_nltk()

    log.info("== Fase 3: Preparación (limpieza + TOP-K autores + 'other' + split 80/10/10) ==")
    (Xtr, ytr), (Xva, yva), (Xte, yte), labels = prepare(df, top_k=8, seed=42)
    log.info("Splits -> train=%d, val=%d, test=%d | labels=%s", len(Xtr), len(Xva), len(Xte), labels)

    log.info("== Fase 4A: Modelado (discriminador TF-IDF + LinearSVC) ==")
    clf = train_discriminator(Xtr, ytr, seed=42)
    log.info("Discriminador entrenado.")

    log.info("== Fase 5A: Evaluación discriminador ==")
    val_metrics = evaluate_classifier(clf, Xva, yva, "val")
    test_metrics = evaluate_classifier(clf, Xte, yte, "test")
    log.info("VAL F1_macro=%.4f | TEST F1_macro=%.4f", val_metrics["f1_macro"], test_metrics["f1_macro"])

    log.info("== Fase 4B: Modelado (generador n-gramas) ==")
    starts, counts = build_ngram_model(Xtr, n=3)
    gen_text = generate_text(starts, counts, n=3, max_tokens=120, seed=42, temperature=0.9)
    log.info("Texto generado (preview): %s", gen_text[:160].replace("\n"," "))

    log.info("== Fase 5B: Evaluación generador (BLEU/ROUGE) ==")
    gen_eval = evaluate_generator(gen_text, refs=list(Xva[:5]))
    log.info("GEN Métricas: %s", gen_eval)

    log.info("== Fase 6: Despliegue (persistencia de los modelos) ==")
    path = save_model(clf, path="artifacts/model_discriminador.joblib")
    log.info("Modelo guardado en: %s", path)   
    path = save_ngram_model((starts, counts, 3), path="artifacts/model_ngram.joblib")
    log.info("Modelo guardado en: %s", path)


    log.info("Pipeline completo ✅")

if __name__ == "__main__":
    main()
