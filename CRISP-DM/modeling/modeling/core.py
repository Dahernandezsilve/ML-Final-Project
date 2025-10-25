import logging
from collections import defaultdict, Counter
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

log = logging.getLogger(__name__)

def _ensure_stopwords():
    try:
        return stopwords.words("spanish")
    except LookupError:
        import nltk
        log.warning("Stopwords 'spanish' no encontradas. Descargando...")
        nltk.download("stopwords")
        return stopwords.words("spanish")

def train_discriminator(Xtr, ytr, seed: int = 42):
    """
    Baseline discriminador: TF-IDF (1-2 gram) + LinearSVC.
    Sin class_weight ni resampling: acorde a tu baseline.
    """
    stop_es = _ensure_stopwords()
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=stop_es,
        ngram_range=(1,2),
        min_df=2
    )
    clf = Pipeline([
        ("tfidf", vec),
        ("clf", LinearSVC(random_state=seed))
    ])
    log.info("Entrenando discriminador TF-IDF + LinearSVC ...")
    clf.fit(Xtr, ytr)
    log.info("Discriminador entrenado.")
    return clf

# --------- Generador n-gramas ----------
def build_ngram_model(texts, n=3):
    log.info("Construyendo modelo n-gramas (n=%d)...", n)
    starts = []
    counts = defaultdict(Counter)
    for doc in texts:
        tokens = str(doc).strip().split()
        if len(tokens) < n:
            continue
        starts.append(tuple(tokens[:n-1]))
        for i in range(len(tokens)-n+1):
            ctx = tuple(tokens[i:i+n-1])
            nxt = tokens[i+n-1]
            counts[ctx][nxt] += 1
    log.info("Modelo n-gramas listo. Contextos: %d", len(counts))
    return starts, counts

def _sample(counter: Counter, temperature=1.0):
    items, freqs = zip(*counter.items())
    p = np.array(freqs, dtype=float) ** (1.0/temperature)
    p /= p.sum()
    return np.random.choice(items, p=p)

def generate_text(starts, counts, n=3, max_tokens=120, seed=42, temperature=0.9):
    log.info("Generando texto (max_tokens=%d, temp=%.2f)...", max_tokens, temperature)
    np.random.seed(seed)
    if not starts:
        log.warning("No hay arranques disponibles (starts vacío).")
        return ""
    import random
    context = list(random.choice(starts))
    out = list(context)
    for _ in range(max_tokens):
        ctx = tuple(context)
        if ctx not in counts or not counts[ctx]:
            break
        nxt = _sample(counts[ctx], temperature)
        out.append(nxt)
        context = (context + [nxt])[-(n-1):]
    text = " ".join(out)
    log.debug("Texto generado (preview 160): %s", text[:160])
    return text

