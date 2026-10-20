"""Generates complete, self-contained Google Colab notebooks for AutoVoice.

Each notebook:
1. Detects Google Colab runtime and prompts the user to upload data (`files.upload()`)
   if the dataset is not already present.
2. Runs end-to-end data processing, training (with GPU where applicable), and evaluation.
3. Automatically bundles all generated models, metrics, and plots into a zip archive
   and triggers browser download (`files.download(...)`) upon completion.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def make_notebook(cells: list[dict], accelerator: str = "None") -> dict:
    """Create a standard Jupyter notebook structure."""
    return {
        "cells": cells,
        "metadata": {
            "accelerator": accelerator,
            "colab": {"provenance": []},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def make_md_cell(source: list[str]) -> dict:
    """Create a markdown notebook cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_code_cell(source: list[str]) -> dict:
    """Create a code notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ===========================================================================
# 01_eda.ipynb: Exploratory Data Analysis & Quality Auditing
# ===========================================================================
eda_cells = [
    make_md_cell(
        [
            "# AutoVoice: Exploratory Data Analysis (NHTSA Complaints)\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devdasrahul/autovoice/blob/main/notebooks/01_eda.ipynb)\n\n",
            "This notebook inspects public NHTSA safety defect complaint records, calculates length distributions, ",
            "evaluates component/make frequencies, and analyzes physical hazard indicators.\n\n",
            "**Google Colab Ready:**\n",
            "- If running in Colab, it will automatically prompt you to upload `complaints.parquet` or `nhtsa_complaints_raw.csv`.\n",
            "- When execution completes, all generated charts and `eda_summary.json` are packaged and **automatically downloaded** to your machine.\n",
        ]
    ),
    make_code_cell(
        [
            "# 1. Environment Setup\n",
            "import os, sys, json, shutil\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "try:\n",
            "    from google.colab import files\n",
            "    IN_COLAB = True\n",
            "    print('🚀 Running inside Google Colab environment.')\n",
            "except ImportError:\n",
            "    IN_COLAB = False\n",
            "    print('💻 Running in standard local Python environment.')\n",
        ]
    ),
    make_code_cell(
        [
            "# 2. Interactive Data Upload / Loading\n",
            "data_file = None\n",
            "candidate_paths = [\n",
            "    Path('complaints.parquet'),\n",
            "    Path('data/processed/complaints.parquet'),\n",
            "    Path('../data/processed/complaints.parquet'),\n",
            "    Path('nhtsa_complaints_raw.csv'),\n",
            "    Path('data/raw/nhtsa_complaints_raw.csv'),\n",
            "]\n",
            "\n",
            "for cp in candidate_paths:\n",
            "    if cp.exists():\n",
            "        data_file = cp\n",
            "        break\n",
            "\n",
            "if data_file is None and IN_COLAB:\n",
            "    print('📁 Data file not found locally. Please upload complaints.parquet or nhtsa_complaints_raw.csv:')\n",
            "    uploaded = files.upload()\n",
            "    for fname in uploaded.keys():\n",
            "        if fname.endswith(('.parquet', '.csv')):\n",
            "            data_file = Path(fname)\n",
            "            break\n",
            "\n",
            "if data_file is None:\n",
            "    raise FileNotFoundError('No complaint data file found. Please upload complaints.parquet or run make data.')\n",
            "\n",
            "print(f'✅ Loading data from: {data_file}')\n",
            "if str(data_file).endswith('.parquet'):\n",
            "    df = pd.read_parquet(data_file)\n",
            "else:\n",
            "    df = pd.read_csv(data_file)\n",
            "    # normalize column names\n",
            "    col_map = {'CDESCR': 'narrative', 'MAKE': 'make', 'MODEL': 'model', 'YEAR': 'model_year', 'COMPNAME': 'component'}\n",
            "    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})\n",
            "\n",
            "text_col = 'narrative' if 'narrative' in df.columns else ('cdescr' if 'cdescr' in df.columns else df.columns[0])\n",
            "print(f'Corpus size: {len(df)} rows. Target text column: {text_col}')\n",
            "df.head(2)\n",
        ]
    ),
    make_code_cell(
        [
            "# 3. Narrative Length Distribution Analysis\n",
            "output_dir = Path('eda_artifacts')\n",
            "output_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "df['char_length'] = df[text_col].fillna('').astype(str).str.len()\n",
            "df['word_count'] = df[text_col].fillna('').astype(str).str.split().str.len()\n",
            "\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
            "df['char_length'].hist(bins=35, ax=ax1, color='#0284c7', edgecolor='black', alpha=0.8)\n",
            "ax1.set_title('Narrative Character Length Distribution')\n",
            "ax1.set_xlabel('Character Count')\n",
            "ax1.set_ylabel('Number of Complaints')\n",
            "\n",
            "df['word_count'].hist(bins=35, ax=ax2, color='#ea580c', edgecolor='black', alpha=0.8)\n",
            "ax2.set_title('Narrative Word Count Distribution')\n",
            "ax2.set_xlabel('Word Count')\n",
            "\n",
            "plt.tight_layout()\n",
            "fig.savefig(output_dir / 'length_distribution.png', dpi=150)\n",
            "plt.show()\n",
            "\n",
            "print(df[['char_length', 'word_count']].describe())\n",
        ]
    ),
    make_code_cell(
        [
            "# 4. Top Vehicle Makes & Defective Components\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "if 'make' in df.columns:\n",
            "    df['make'].value_counts().head(10).plot(kind='barh', ax=ax1, color='#3b82f6')\n",
            "    ax1.set_title('Top 10 Reported Vehicle Makes')\n",
            "    ax1.invert_yaxis()\n",
            "\n",
            "if 'component' in df.columns:\n",
            "    df['component'].value_counts().head(10).plot(kind='barh', ax=ax2, color='#10b981')\n",
            "    ax2.set_title('Top 10 Reported Defect Components')\n",
            "    ax2.invert_yaxis()\n",
            "\n",
            "plt.tight_layout()\n",
            "fig.savefig(output_dir / 'top_makes_and_components.png', dpi=150)\n",
            "plt.show()\n",
        ]
    ),
    make_code_cell(
        [
            "# 5. Physical Hazard Indicators Analysis\n",
            "hazard_cols = [c for c in ['crash', 'fire', 'injured', 'deaths'] if c in df.columns]\n",
            "if hazard_cols:\n",
            "    hazard_counts = df[hazard_cols].sum()\n",
            "    print('Hazard Indicator Counts:\\n', hazard_counts)\n",
            "    \n",
            "    fig, ax = plt.subplots(figsize=(7, 4))\n",
            "    hazard_counts.plot(kind='bar', ax=ax, color=['#ef4444', '#f97316', '#eab308', '#dc2626'])\n",
            "    ax.set_title('Severe Safety Hazard Incidents in Fleet')\n",
            "    ax.set_ylabel('Number of Reported Incidents')\n",
            "    plt.tight_layout()\n",
            "    fig.savefig(output_dir / 'safety_hazards.png', dpi=150)\n",
            "    plt.show()\n",
        ]
    ),
    make_code_cell(
        [
            "# 6. Generate and Save Summary Report\n",
            "summary = {\n",
            "    'total_records': len(df),\n",
            "    'mean_char_length': round(float(df['char_length'].mean()), 2),\n",
            "    'median_word_count': round(float(df['word_count'].median()), 2),\n",
            "    'top_components': df['component'].value_counts().head(5).to_dict() if 'component' in df.columns else {},\n",
            "    'hazard_counts': {c: int(df[c].sum()) for c in hazard_cols},\n",
            "}\n",
            "\n",
            "(output_dir / 'eda_summary.json').write_text(json.dumps(summary, indent=2))\n",
            "print('Summary saved:', json.dumps(summary, indent=2))\n",
        ]
    ),
    make_code_cell(
        [
            "# 7. Automated Output Packaging & Browser Download\n",
            "archive_name = 'autovoice_eda_artifacts'\n",
            "shutil.make_archive(archive_name, 'zip', output_dir)\n",
            "print(f'📦 Packaged EDA outputs into {archive_name}.zip')\n",
            "\n",
            "if IN_COLAB:\n",
            "    print('⬇️ Downloading artifact archive to your computer...')\n",
            "    files.download(f'{archive_name}.zip')\n",
            "else:\n",
            "    print(f'Artifact archive available at: {Path(archive_name + \".zip\").resolve()}')\n",
        ]
    ),
]


# ===========================================================================
# 02_train_ner_colab.ipynb: DistilBERT Token Classification
# ===========================================================================
ner_cells = [
    make_md_cell(
        [
            "# AutoVoice: Train DistilBERT Token Classification (NER) on GPU\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devdasrahul/autovoice/blob/main/notebooks/02_train_ner_colab.ipynb)\n\n",
            "This notebook fine-tunes `distilbert-base-uncased` for sequence token labeling to extract:\n",
            "- `COMPONENT`: Automotive subsystems and parts (e.g. *steering rack*, *master cylinder*, *airbag*)\n",
            "- `SYMPTOM`: Failure behaviors and defects (e.g. *stalled*, *leaking fluid*, *smoke*, *locked up*)\n\n",
            "**Google Colab Ready:**\n",
            "- Select **Runtime > Change runtime type > T4 GPU**.\n",
            "- Prompts to upload `complaints.parquet` if not present.\n",
            "- When training finishes, automatically zips and downloads `autovoice_ner_model_artifacts.zip` to your computer.\n",
        ]
    ),
    make_code_cell(
        [
            "# 1. Environment Setup & Dependency Installation\n",
            "!pip install -q transformers==4.40.0 datasets seqeval accelerate torch\n",
            "\n",
            "import os, sys, json, shutil, re\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import torch\n",
            "\n",
            "try:\n",
            "    from google.colab import files\n",
            "    IN_COLAB = True\n",
            "    print('🚀 Running in Google Colab.')\n",
            "except ImportError:\n",
            "    IN_COLAB = False\n",
            "    print('💻 Running locally.')\n",
            "\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print(f'Hardware Accelerator: {device}')\n",
            "if device == 'cuda':\n",
            "    print(f'GPU Device: {torch.cuda.get_device_name(0)}')\n",
        ]
    ),
    make_code_cell(
        [
            "# 2. Interactive Data Loading\n",
            "data_file = None\n",
            "candidate_paths = [\n",
            "    Path('complaints.parquet'),\n",
            "    Path('data/processed/complaints.parquet'),\n",
            "    Path('../data/processed/complaints.parquet'),\n",
            "    Path('nhtsa_complaints_raw.csv'),\n",
            "]\n",
            "\n",
            "for cp in candidate_paths:\n",
            "    if cp.exists():\n",
            "        data_file = cp\n",
            "        break\n",
            "\n",
            "if data_file is None and IN_COLAB:\n",
            "    print('📁 Please upload complaints.parquet or nhtsa_complaints_raw.csv:')\n",
            "    uploaded = files.upload()\n",
            "    for fname in uploaded.keys():\n",
            "        if fname.endswith(('.parquet', '.csv')):\n",
            "            data_file = Path(fname)\n",
            "            break\n",
            "\n",
            "if data_file is None:\n",
            "    raise FileNotFoundError('No complaint dataset found. Please upload complaints.parquet.')\n",
            "\n",
            "print(f'✅ Loading dataset from {data_file}')\n",
            "df = pd.read_parquet(data_file) if str(data_file).endswith('.parquet') else pd.read_csv(data_file)\n",
            "text_col = 'narrative' if 'narrative' in df.columns else ('cdescr' if 'cdescr' in df.columns else df.columns[0])\n",
            "texts = df[text_col].dropna().astype(str).tolist()\n",
            "print(f'Loaded {len(texts)} complaint narratives.')\n",
        ]
    ),
    make_code_cell(
        [
            "# 3. Self-Contained BIO Schema & Weak Labeling Engine\n",
            "LABELS = ['O', 'B-COMPONENT', 'I-COMPONENT', 'B-SYMPTOM', 'I-SYMPTOM']\n",
            "label2id = {l: i for i, l in enumerate(LABELS)}\n",
            "id2label = {i: l for i, l in enumerate(LABELS)}\n",
            "\n",
            "# Curated automotive lexicons for weak silver labeling\n",
            "COMPONENTS = ['steering', 'power steering', 'brakes', 'brake', 'engine', 'transmission', 'airbag', 'air bag', 'suspension', 'battery', 'accelerator', 'throttle', 'tire', 'seat belt', 'fuel pump', 'alternator']\n",
            "SYMPTOMS = ['stalled', 'stall', 'shut off', 'died', 'won\\'t start', 'failed to start', 'fire', 'caught fire', 'smoke', 'smoking', 'crash', 'collision', 'locked up', 'vibration', 'shaking', 'leaking', 'leak', 'loss of power', 'warning light', 'loss of control']\n",
            "\n",
            "def tag_text(text):\n",
            "    words = text.split()\n",
            "    tags = ['O'] * len(words)\n",
            "    text_lower = text.lower()\n",
            "    # Match symptoms\n",
            "    for sym in SYMPTOMS:\n",
            "        for m in re.finditer(rf'\\b{re.escape(sym)}\\b', text_lower):\n",
            "            start_char = m.start()\n",
            "            end_char = m.end()\n",
            "            curr_char = 0\n",
            "            for i, w in enumerate(words):\n",
            "                w_start = text.find(w, curr_char)\n",
            "                w_end = w_start + len(w)\n",
            "                curr_char = w_end\n",
            "                if w_start >= start_char and w_end <= end_char:\n",
            "                    tags[i] = 'B-SYMPTOM' if tags[i] == 'O' and (i == 0 or 'SYMPTOM' not in tags[i-1]) else 'I-SYMPTOM'\n",
            "    # Match components\n",
            "    for comp in COMPONENTS:\n",
            "        for m in re.finditer(rf'\\b{re.escape(comp)}\\b', text_lower):\n",
            "            start_char = m.start()\n",
            "            end_char = m.end()\n",
            "            curr_char = 0\n",
            "            for i, w in enumerate(words):\n",
            "                w_start = text.find(w, curr_char)\n",
            "                w_end = w_start + len(w)\n",
            "                curr_char = w_end\n",
            "                if w_start >= start_char and w_end <= end_char and tags[i] == 'O':\n",
            "                    tags[i] = 'B-COMPONENT' if (i == 0 or 'COMPONENT' not in tags[i-1]) else 'I-COMPONENT'\n",
            "    return words, [label2id[t] for t in tags]\n",
            "\n",
            "sample_words, sample_tags = tag_text('The power steering failed at 65 mph causing loss of control and crash.')\n",
            "for w, t in zip(sample_words, sample_tags):\n",
            "    if t != 0:\n",
            "        print(f'{w} -> {id2label[t]}')\n",
        ]
    ),
    make_code_cell(
        [
            "# 4. Build Dataset & Tokenizer Alignment\n",
            "from transformers import AutoTokenizer\n",
            "from datasets import Dataset\n",
            "\n",
            "MODEL_ID = 'distilbert-base-uncased'\n",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n",
            "\n",
            "all_tokens, all_ner_tags = [], []\n",
            "for t in texts:\n",
            "    words, tags = tag_text(t)\n",
            "    if words:\n",
            "        all_tokens.append(words)\n",
            "        all_ner_tags.append(tags)\n",
            "\n",
            "raw_dataset = Dataset.from_dict({'tokens': all_tokens, 'ner_tags': all_ner_tags})\n",
            "split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)\n",
            "\n",
            "def tokenize_and_align_labels(examples):\n",
            "    tokenized = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, max_length=256)\n",
            "    labels = []\n",
            "    for i, label in enumerate(examples['ner_tags']):\n",
            "        word_ids = tokenized.word_ids(batch_index=i)\n",
            "        previous_word_idx = None\n",
            "        label_ids = []\n",
            "        for word_idx in word_ids:\n",
            "            if word_idx is None:\n",
            "                label_ids.append(-100)\n",
            "            elif word_idx != previous_word_idx:\n",
            "                label_ids.append(label[word_idx] if word_idx < len(label) else -100)\n",
            "            else:\n",
            "                # Inside same word sub-token\n",
            "                cur_label = label[word_idx] if word_idx < len(label) else -100\n",
            "                label_ids.append(cur_label)\n",
            "            previous_word_idx = word_idx\n",
            "        labels.append(label_ids)\n",
            "    tokenized['labels'] = labels\n",
            "    return tokenized\n",
            "\n",
            "train_ds = split_dataset['train'].map(tokenize_and_align_labels, batched=True)\n",
            "eval_ds = split_dataset['test'].map(tokenize_and_align_labels, batched=True)\n",
            "print(f'Train examples: {len(train_ds)}, Test examples: {len(eval_ds)}')\n",
        ]
    ),
    make_code_cell(
        [
            "# 5. Fine-Tune DistilBERT with seqeval Evaluation\n",
            "from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification\n",
            "import evaluate\n",
            "\n",
            "seqeval_metric = evaluate.load('seqeval')\n",
            "\n",
            "def compute_metrics(p):\n",
            "    predictions, labels = p\n",
            "    predictions = np.argmax(predictions, axis=2)\n",
            "    true_predictions = [\n",
            "        [id2label[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]\n",
            "        for prediction, label in zip(predictions, labels)\n",
            "    ]\n",
            "    true_labels = [\n",
            "        [id2label[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]\n",
            "        for prediction, label in zip(predictions, labels)\n",
            "    ]\n",
            "    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)\n",
            "    return {\n",
            "        'precision': results['overall_precision'],\n",
            "        'recall': results['overall_recall'],\n",
            "        'f1': results['overall_f1'],\n",
            "        'accuracy': results['overall_accuracy'],\n",
            "    }\n",
            "\n",
            "model = AutoModelForTokenClassification.from_pretrained(\n",
            "    MODEL_ID,\n",
            "    num_labels=len(LABELS),\n",
            "    id2label=id2label,\n",
            "    label2id=label2id,\n",
            ")\n",
            "\n",
            "training_args = TrainingArguments(\n",
            "    output_dir='./results_ner',\n",
            "    evaluation_strategy='epoch',\n",
            "    save_strategy='epoch',\n",
            "    learning_rate=3e-5,\n",
            "    per_device_train_batch_size=16,\n",
            "    per_device_eval_batch_size=16,\n",
            "    num_train_epochs=3,\n",
            "    weight_decay=0.01,\n",
            "    load_best_model_at_end=True,\n",
            "    metric_for_best_model='f1',\n",
            "    report_to='none',\n",
            ")\n",
            "\n",
            "trainer = Trainer(\n",
            "    model=model,\n",
            "    args=training_args,\n",
            "    train_dataset=train_ds,\n",
            "    eval_dataset=eval_ds,\n",
            "    tokenizer=tokenizer,\n",
            "    data_collator=DataCollatorForTokenClassification(tokenizer),\n",
            "    compute_metrics=compute_metrics,\n",
            ")\n",
            "\n",
            "print('🚀 Commencing DistilBERT training...')\n",
            "trainer.train()\n",
        ]
    ),
    make_code_cell(
        [
            "# 6. Evaluate Model on Test Split\n",
            "eval_results = trainer.evaluate()\n",
            "print('Verified NER Evaluation Metrics:', json.dumps(eval_results, indent=2))\n",
            "\n",
            "export_dir = Path('autovoice_ner_model')\n",
            "export_dir.mkdir(parents=True, exist_ok=True)\n",
            "trainer.save_model(export_dir)\n",
            "tokenizer.save_pretrained(export_dir)\n",
            "(export_dir / 'ner_metrics.json').write_text(json.dumps(eval_results, indent=2))\n",
            "print('Saved model and metrics to:', export_dir)\n",
        ]
    ),
    make_code_cell(
        [
            "# 7. Interactive Sample Inference\n",
            "from transformers import pipeline\n",
            "\n",
            "ner_pipe = pipeline('ner', model=str(export_dir), tokenizer=str(export_dir), aggregation_strategy='simple')\n",
            "sample_narrative = 'While driving at 65 mph, the power steering suddenly locked up, resulting in brake failure and a crash.'\n",
            "print('Inference Sample:', sample_narrative)\n",
            "extracted = ner_pipe(sample_narrative)\n",
            "for ent in extracted:\n",
            "    print(f'Entity: {ent[\"word\"]} -> {ent[\"entity_group\"]} (Score: {ent[\"score\"]:.4f})')\n",
        ]
    ),
    make_code_cell(
        [
            "# 8. Automated Model Artifact Packaging & Browser Download\n",
            "archive_name = 'autovoice_ner_model_artifacts'\n",
            "shutil.make_archive(archive_name, 'zip', export_dir)\n",
            "print(f'📦 Packaged DistilBERT NER model artifacts into {archive_name}.zip')\n",
            "\n",
            "if IN_COLAB:\n",
            "    print('⬇️ Downloading model artifacts to your local machine...')\n",
            "    files.download(f'{archive_name}.zip')\n",
            "else:\n",
            "    print(f'Archive created at: {Path(archive_name + \".zip\").resolve()}')\n",
        ]
    ),
]


# ===========================================================================
# 03_train_classifier_colab.ipynb: Severity Classifier Triage
# ===========================================================================
clf_cells = [
    make_md_cell(
        [
            "# AutoVoice: Train Severity Classifier (Triage Prioritization)\n",
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devdasrahul/autovoice/blob/main/notebooks/03_train_classifier_colab.ipynb)\n\n",
            "This notebook trains the AutoVoice triage severity classifier on NHTSA complaint narratives.\n",
            "- **Architecture:** Sublinear TF-IDF (1-2 word n-grams) + Class-Weighted Logistic Regression (`class_weight='balanced'`)\n",
            "- **Proxy Severity Labels:** Derived strictly from objective indicator fields (`deaths`, `injured`, `fire`, `crash`) with zero narrative text leakage.\n",
            "- **Triage Classes:** `LOW`, `MEDIUM`, `HIGH`\n\n",
            "**Google Colab Ready:**\n",
            "- Prompts to upload `complaints.parquet` if not already present.\n",
            "- When training completes, confusion matrix plot and model artifacts (`model.pkl`, `vectorizer.pkl`) are **automatically downloaded** to your machine.\n",
        ]
    ),
    make_code_cell(
        [
            "# 1. Environment Setup\n",
            "!pip install -q scikit-learn pandas numpy pyarrow matplotlib seaborn\n",
            "\n",
            "import os, sys, json, pickle, shutil\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "try:\n",
            "    from google.colab import files\n",
            "    IN_COLAB = True\n",
            "    print('🚀 Running in Google Colab.')\n",
            "except ImportError:\n",
            "    IN_COLAB = False\n",
            "    print('💻 Running locally.')\n",
        ]
    ),
    make_code_cell(
        [
            "# 2. Interactive Data Upload / Loading\n",
            "data_file = None\n",
            "candidate_paths = [\n",
            "    Path('complaints.parquet'),\n",
            "    Path('data/processed/complaints.parquet'),\n",
            "    Path('../data/processed/complaints.parquet'),\n",
            "    Path('nhtsa_complaints_raw.csv'),\n",
            "]\n",
            "\n",
            "for cp in candidate_paths:\n",
            "    if cp.exists():\n",
            "        data_file = cp\n",
            "        break\n",
            "\n",
            "if data_file is None and IN_COLAB:\n",
            "    print('📁 Please upload complaints.parquet or nhtsa_complaints_raw.csv:')\n",
            "    uploaded = files.upload()\n",
            "    for fname in uploaded.keys():\n",
            "        if fname.endswith(('.parquet', '.csv')):\n",
            "            data_file = Path(fname)\n",
            "            break\n",
            "\n",
            "if data_file is None:\n",
            "    raise FileNotFoundError('No complaint dataset found. Please upload complaints.parquet.')\n",
            "\n",
            "print(f'✅ Loading dataset from {data_file}')\n",
            "df = pd.read_parquet(data_file) if str(data_file).endswith('.parquet') else pd.read_csv(data_file)\n",
            "text_col = 'narrative' if 'narrative' in df.columns else ('cdescr' if 'cdescr' in df.columns else df.columns[0])\n",
            "\n",
            "# Derive severity labels from physical indicators if not already computed\n",
            "if 'severity' not in df.columns:\n",
            "    def derive_severity(row):\n",
            "        deaths = int(row.get('deaths', 0) or 0)\n",
            "        injured = int(row.get('injured', 0) or 0)\n",
            "        fire = bool(row.get('fire', False))\n",
            "        crash = bool(row.get('crash', False))\n",
            "        if deaths > 0 or injured > 0 or fire:\n",
            "            return 'HIGH'\n",
            "        if crash:\n",
            "            return 'MEDIUM'\n",
            "        return 'LOW'\n",
            "    df['severity'] = df.apply(derive_severity, axis=1)\n",
            "\n",
            "print('Class distribution:\\n', df['severity'].value_counts())\n",
        ]
    ),
    make_code_cell(
        [
            "# 3. Train / Test Split & Feature Extraction\n",
            "X = df[text_col].fillna('').astype(str).tolist()\n",
            "y = df['severity'].tolist()\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "print(f'Train samples: {len(X_train)}, Test samples: {len(X_test)}')\n",
            "\n",
            "vectorizer = TfidfVectorizer(\n",
            "    ngram_range=(1, 2),\n",
            "    max_features=5000,\n",
            "    sublinear_tf=True,\n",
            "    strip_accents='unicode',\n",
            "    lowercase=True,\n",
            ")\n",
            "\n",
            "X_train_vec = vectorizer.fit_transform(X_train)\n",
            "X_test_vec = vectorizer.transform(X_test)\n",
            "print(f'Vocabulary size: {len(vectorizer.vocabulary_)} features')\n",
        ]
    ),
    make_code_cell(
        [
            "# 4. Train Class-Weighted Logistic Regression Model\n",
            "clf = LogisticRegression(\n",
            "    class_weight='balanced',\n",
            "    max_iter=1000,\n",
            "    C=1.0,\n",
            "    random_state=42,\n",
            ")\n",
            "\n",
            "print('Training severity classifier...')\n",
            "clf.fit(X_train_vec, y_train)\n",
            "print('Training complete!')\n",
        ]
    ),
    make_code_cell(
        [
            "# 5. Evaluation & Confusion Matrix Heatmap\n",
            "output_dir = Path('classifier_artifacts')\n",
            "output_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "y_pred = clf.predict(X_test_vec)\n",
            "acc = accuracy_score(y_test, y_pred)\n",
            "weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)\n",
            "macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)\n",
            "\n",
            "print(f'Accuracy: {acc*100:.2f}%')\n",
            "print(f'Weighted F1: {weighted_f1:.4f}')\n",
            "print(f'Macro F1: {macro_f1:.4f}\\n')\n",
            "print('Classification Report:\\n', classification_report(y_test, y_pred, zero_division=0))\n",
            "\n",
            "classes = sorted(list(set(y)))\n",
            "cm = confusion_matrix(y_test, y_pred, labels=classes)\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(6, 5))\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)\n",
            "ax.set_title('Triage Severity Confusion Matrix')\n",
            "ax.set_xlabel('Predicted Class')\n",
            "ax.set_ylabel('True Indicator Class')\n",
            "plt.tight_layout()\n",
            "fig.savefig(output_dir / 'confusion_matrix.png', dpi=150)\n",
            "plt.show()\n",
        ]
    ),
    make_code_cell(
        [
            "# 6. Export Model, Vectorizer & Metrics\n",
            "with open(output_dir / 'model.pkl', 'wb') as f:\n",
            "    pickle.dump(clf, f)\n",
            "with open(output_dir / 'vectorizer.pkl', 'wb') as f:\n",
            "    pickle.dump(vectorizer, f)\n",
            "\n",
            "metrics = {\n",
            "    'accuracy': round(float(acc), 4),\n",
            "    'weighted_f1': round(float(weighted_f1), 4),\n",
            "    'macro_f1': round(float(macro_f1), 4),\n",
            "    'classes': classes,\n",
            "}\n",
            "(output_dir / 'clf_metrics.json').write_text(json.dumps(metrics, indent=2))\n",
            "print('Saved model and metrics to:', output_dir)\n",
        ]
    ),
    make_code_cell(
        [
            "# 7. Interactive Triage Inference Sample\n",
            "test_samples = [\n",
            "    'Vehicle experienced a minor cosmetic scratch on plastic hubcap while parked.',\n",
            "    'Brakes failed at 50 mph resulting in rear-end crash with guardrail.',\n",
            "    'High-voltage battery caught fire on highway. Ambulance called for driver burns.',\n",
            "]\n",
            "\n",
            "for sample in test_samples:\n",
            "    vec = vectorizer.transform([sample])\n",
            "    pred = clf.predict(vec)[0]\n",
            "    probs = {cls_name: round(float(p), 4) for cls_name, p in zip(clf.classes_, clf.predict_proba(vec)[0])}\n",
            "    print(f'Text: \"{sample}\"')\n",
            "    print(f'-> Predicted Severity: {pred}')\n",
            "    print(f'-> Calibrated Distribution: {probs}\\n')\n",
        ]
    ),
    make_code_cell(
        [
            "# 8. Automated Output Packaging & Browser Download\n",
            "archive_name = 'autovoice_classifier_artifacts'\n",
            "shutil.make_archive(archive_name, 'zip', output_dir)\n",
            "print(f'📦 Packaged Classifier model artifacts into {archive_name}.zip')\n",
            "\n",
            "if IN_COLAB:\n",
            "    print('⬇️ Downloading model artifacts to your local machine...')\n",
            "    files.download(f'{archive_name}.zip')\n",
            "else:\n",
            "    print(f'Archive created at: {Path(archive_name + \".zip\").resolve()}')\n",
        ]
    ),
]


def generate_all_notebooks():
    """Write all 3 notebooks to notebooks/."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    eda_p = NOTEBOOKS_DIR / "01_eda.ipynb"
    eda_p.write_text(json.dumps(make_notebook(eda_cells, accelerator="None"), indent=1), encoding="utf-8")
    print(f"Generated {eda_p}")

    ner_p = NOTEBOOKS_DIR / "02_train_ner_colab.ipynb"
    ner_p.write_text(json.dumps(make_notebook(ner_cells, accelerator="GPU"), indent=1), encoding="utf-8")
    print(f"Generated {ner_p}")

    clf_p = NOTEBOOKS_DIR / "03_train_classifier_colab.ipynb"
    clf_p.write_text(json.dumps(make_notebook(clf_cells, accelerator="None"), indent=1), encoding="utf-8")
    print(f"Generated {clf_p}")


if __name__ == "__main__":
    generate_all_notebooks()
