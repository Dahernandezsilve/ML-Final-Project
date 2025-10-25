import logging
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from nltk.tokenize import word_tokenize

log = logging.getLogger(__name__)

def _ensure_punkt():
    try:
        word_tokenize("hola")
    except LookupError:
        import nltk
        log.warning("Punkt no encontrado. Descargando...")
        nltk.download("punkt")

def evaluate_classifier(model, X, y, split_name="val") -> dict:
    log.info("Evaluando discriminador en split=%s...", split_name)
    yhat = model.predict(X)
    report = classification_report(y, yhat, digits=3, output_dict=True)
    f1m = f1_score(y, yhat, average="macro")
    out = {"split": split_name, "f1_macro": float(f1m), "report": report}
    log.info("F1_macro(%s)=%.4f", split_name, f1m)
    return out

def confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    log.debug("Matriz de confusión shape=%s", cm.shape)
    return cm.tolist()

# -------- Evaluación generador (BLEU/ROUGE) ----------
def _safe_bleu(hyp_tokens, refs_tokens):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method4
        return float(sentence_bleu(refs_tokens, hyp_tokens, smoothing_function=smoothie)) # type: ignore
    except Exception as e:
        log.warning("BLEU no disponible (%s). Devuelvo 0.", e)
        return 0.0

def _rouge_f1(gen_text, refs):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        scores = [scorer.score(r, gen_text) for r in refs if isinstance(r, str)]
        if not scores:
            return {"ROUGE1_F1":0.0,"ROUGE2_F1":0.0,"ROUGEL_F1":0.0}
        r1 = float(np.mean([s["rouge1"].fmeasure for s in scores]))
        r2 = float(np.mean([s["rouge2"].fmeasure for s in scores]))
        rl = float(np.mean([s["rougeL"].fmeasure for s in scores]))
        return {"ROUGE1_F1": r1, "ROUGE2_F1": r2, "ROUGEL_F1": rl}
    except Exception as e:
        log.warning("ROUGE no disponible (%s). Devolviendo 0s.", e)
        return {"ROUGE1_F1":0.0,"ROUGE2_F1":0.0,"ROUGEL_F1":0.0}

def evaluate_generator(gen_text: str, refs: list[str]):
    log.info("Evaluando generador (BLEU/ROUGE) ...")
    _ensure_punkt()
    hyp_tokens = word_tokenize(str(gen_text))
    refs_tokens = [word_tokenize(str(r)) for r in refs if isinstance(r, str)]
    bleu = _safe_bleu(hyp_tokens, refs_tokens) if refs_tokens else 0.0
    rouge = _rouge_f1(gen_text, refs)
    out = {"BLEU": bleu, **rouge}
    log.info("GEN -> %s", out)
    return out
