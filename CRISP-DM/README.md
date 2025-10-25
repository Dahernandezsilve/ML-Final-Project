# CRISP-DM Project (UVG — NLP)
Implementación del ciclo CRISP-DM con **dos modelos antagonistas**:
- **Generador** (n-gramas) que produce texto poético al estilo del Siglo de Oro.
- **Discriminador** (TF-IDF + LinearSVC) que decide si un poema es auténtico o generado.

**Balanceo**: selección de **TOP-K autores** y **agregación del resto en `other`** (sin up/down-sampling), con split **80/10/10 estratificado**.

## Requisitos
- Python 3.9+
- Internet para descargar dataset HF y corpora NLTK (stopwords/punkt) al primer uso.

## Instalación (modo editable)
cd business_understanding && pip install -e . && cd ..
cd data_understanding   && pip install -e . && cd ..
cd data_preparation     && pip install -e . && cd ..
cd modeling             && pip install -e . && cd ..
cd evaluation           && pip install -e . && cd ..
cd deployment           && pip install -e . && cd ..

## Ejecución
python run_pipeline.py

## Datos
Por defecto se carga el dataset HF `biglam/spanish_golden_age_sonnets`.
