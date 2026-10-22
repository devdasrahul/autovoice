---
language:
- en
license: mit
library_name: transformers
tags:
- token-classification
- ner
- named-entity-recognition
- automotive
- vehicle-safety
- nhtsa
- distilbert
datasets:
- custom
metrics:
- f1
- accuracy
pipeline_tag: token-classification
widget:
- text: "While traveling at 65 mph on the highway, the steering wheel locked up and the vehicle lost power steering. Service engine warning light illuminated and vehicle stalled."
  example_title: "Steering Assist Failure"
- text: "The driver side airbag failed to deploy during a frontal collision at 40 mph, causing severe injuries and smoke from the steering column."
  example_title: "Airbag Deployment Failure"
- text: "Smoke and chemical odor emanated from the high voltage propulsion battery compartment while charging in the garage."
  example_title: "Propulsion Battery Smoke"
model-index:
- name: autovoice-automotive-ner
  results:
  - task:
      type: token-classification
      name: Named Entity Recognition
    metrics:
    - name: Macro F1
      type: f1
      value: 0.9143
    - name: Token Accuracy
      type: accuracy
      value: 0.9817
---

# AutoVoice: Automotive Safety Defect Named Entity Recognition (NER)

**AutoVoice-Automotive-NER** is a fine-tuned `distilbert-base-uncased` token classification model designed to extract automotive components and failure symptoms from unstructured vehicle defect and consumer complaint narratives (such as NHTSA ODI filings).

This model is the core extraction engine of the [AutoVoice](https://github.com/devdasrahul/autovoice) intelligence platform.

---

## Model Details

- **Model Type:** Token Classification (NER)
- **Base Architecture:** `distilbert-base-uncased` (6 layers, 768 hidden dimension, 12 attention heads)
- **Fine-Tuning:** Trained on vehicle defect complaint sequences with 5-class BIO tagging:
  - `O`: Outside of any named entity
  - `B-COMPONENT`: Beginning of an automotive component/subsystem (e.g. *steering wheel*, *airbag*, *propulsion battery*, *transmission*)
  - `I-COMPONENT`: Inside/continuation of an automotive component span
  - `B-SYMPTOM`: Beginning of a failure symptom or behavior (e.g. *locked up*, *stalled*, *smoke*, *loss of power*)
  - `I-SYMPTOM`: Inside/continuation of a failure symptom span
- **Parameters:** ~66.3 Million parameters
- **Format:** PyTorch / SafeTensors (`model.safetensors`, 265 MB)

---

## Evaluation Results

Evaluated on NHTSA vehicle complaint test splits:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **NER Macro F1** | **0.9143** | Macro-averaged F1 across BIO entity spans |
| **Component F1** | **0.8286** | Exact span precision/recall for vehicle components |
| **Symptom F1** | **1.0000** | Exact span precision/recall for failure symptoms |
| **Token Accuracy** | **98.17%** | Token-level classification accuracy on test sequences |
| **CPU Latency (p50)** | **< 20 ms** | Fast CPU inference suitable for real-time web services |

---

## Quickstart & Usage

### Using Hugging Face `transformers` Pipeline

```python
from transformers import pipeline

# Load token classification pipeline
ner_pipeline = pipeline(
    "token-classification",
    model="YOUR_USERNAME/autovoice-automotive-ner",
    aggregation_strategy="simple"  # Combines subword tokens into full entity spans
)

text = (
    "While traveling at 65 mph on the highway, the steering wheel locked up "
    "and the vehicle lost power steering. Service engine warning light illuminated and vehicle stalled."
)

entities = ner_pipeline(text)
for entity in entities:
    print(f"[{entity['entity_group']}] {entity['word']} (score: {entity['score']:.3f})")
```

**Output:**
```text
[COMPONENT] steering wheel (score: 0.992)
[SYMPTOM] locked up (score: 0.985)
[COMPONENT] power steering (score: 0.978)
[COMPONENT] engine (score: 0.965)
[SYMPTOM] warning light (score: 0.954)
[SYMPTOM] stalled (score: 0.991)
```

### Direct Model & Tokenizer Loading

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "YOUR_USERNAME/autovoice-automotive-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

inputs = tokenizer("Airbag failed to deploy during collision.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predictions = torch.argmax(logits, dim=2)
id2label = model.config.id2label

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for token, pred in zip(tokens, predictions[0]):
    print(f"{token:15} -> {id2label[pred.item()]}")
```

---

## Intended Use & Limitations

- **Intended Use:** Automated extraction of vehicle parts and defect behaviors for automotive OEMs, safety engineering teams, fleet telematics, and insurance defect analysis.
- **Language:** English vehicle complaint records.
- **Disclaimer:** Complaint narratives represent unverified owner allegations filed with the NHTSA Office of Defects Investigation. They do not constitute official safety defect findings or manufacturer recall determinations.

---

## Related Projects

- **AutoVoice GitHub Repository:** [github.com/devdasrahul/autovoice](https://github.com/devdasrahul/autovoice)
- **Live Interactive Demo:** [autovoice.streamlit.app](https://autovoice.streamlit.app)
