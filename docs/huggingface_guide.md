# Hugging Face Deployment & Model Publishing Guide

This guide details **what** to upload to Hugging Face from AutoVoice and **how** to upload it step-by-step.

---

## 1. What to Upload to Hugging Face

There are two primary assets you can publish to Hugging Face:

### Asset 1 (Primary): The Fine-Tuned DistilBERT Automotive NER Model
- **Local Path:** `models/ner/`
- **Files to Upload:**
  - `model.safetensors` (265 MB — the fine-tuned DistilBERT weights)
  - `config.json` (Architecture and label mappings: `B-COMPONENT`, `I-COMPONENT`, `B-SYMPTOM`, `I-SYMPTOM`)
  - `tokenizer.json` (Pre-trained WordPiece vocabulary and tokenizer state)
  - `tokenizer_config.json` (Special tokens and casing configuration)
  - `ner_metrics.json` (Evaluation results: 0.9143 Macro F1, 98.17% Token Accuracy)
  - `README.md` (Prepared Hugging Face Model Card with YAML tags, description, and code examples)
- **Hugging Face Destination:** `https://huggingface.co/<your-username>/autovoice-automotive-ner`
- **Why Upload This:**
  - It publishes your fine-tuned model to the global Hugging Face Model Hub.
  - Adds an interactive **Inference Widget** directly on your Hugging Face model page where visitors can test complaints live in their browser.
  - Allows anyone to load your model in 2 lines of Python:
    ```python
    from transformers import pipeline
    ner = pipeline("token-classification", model="<your-username>/autovoice-automotive-ner")
    ```

---

### Asset 2 (Bonus): Hugging Face Space (Streamlit Interactive App)
- **Type:** Hugging Face Space (Streamlit SDK, Free CPU Tier)
- **Hugging Face Destination:** `https://huggingface.co/spaces/<your-username>/autovoice`
- **Why Upload This:**
  - Gives you a permanent secondary live web demo hosted on Hugging Face's global infrastructure alongside your Streamlit Community Cloud deployment.

---

## 2. How to Upload the NER Model to Hugging Face

### Prerequisites
1. Create a free account at [huggingface.co](https://huggingface.co/join) if you haven't already.
2. Generate an Access Token:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Click **Create new token**.
   - Name it `autovoice-upload`, select **Write** permission, and copy the token (`hf_...`).

---

### Method 1: Automated Script (Fastest & Recommended)

AutoVoice includes an automated upload script in `scripts/upload_to_huggingface.py`.

```bash
# 1. Login to Hugging Face using your write token
uv run --with huggingface_hub huggingface-cli login
# (Paste your token when prompted)

# 2. Upload the model (replace with your actual Hugging Face username)
uv run --with huggingface_hub python scripts/upload_to_huggingface.py --repo-id devdasrahul/autovoice-automotive-ner
```

*Alternatively, pass the token as an environment variable or CLI argument:*
```bash
uv run --with huggingface_hub python scripts/upload_to_huggingface.py \
  --repo-id devdasrahul/autovoice-automotive-ner \
  --token hf_YOUR_WRITE_TOKEN
```

The script will automatically create the repository on Hugging Face, upload all model weights, tokenizer files, and the official Model Card.

---

### Method 2: Web Browser Drag-and-Drop (Zero Terminal Commands)

If you prefer using the Hugging Face website interface:

1. **Create Model Repository:**
   - Go to [https://huggingface.co/new](https://huggingface.co/new).
   - **Owner:** Select your username (e.g. `devdasrahul`).
   - **Model name:** `autovoice-automotive-ner`.
   - **License:** `mit`.
   - **Visibility:** Public.
   - Click **Create model**.

2. **Upload Model Files:**
   - On your newly created model page, click on the **Files and versions** tab.
   - Click **Add file** -> **Upload files**.
   - Open your local folder: `models/ner/`.
   - Drag and drop the following files into the browser:
     - `config.json`
     - `model.safetensors`
     - `tokenizer.json`
     - `tokenizer_config.json`
     - `ner_metrics.json`
     - `README.md`
   - In the "Commit changes" message box, type: `Initial upload of AutoVoice DistilBERT NER model`.
   - Click **Commit changes to main**.

3. **Verify:**
   - Click the **Model card** tab. The live inference widget and evaluation metrics will immediately appear!

---

### Method 3: Standard Git & Git LFS

```bash
# 1. Ensure git-lfs is installed
git lfs install

# 2. Clone your empty Hugging Face model repository
git clone https://huggingface.co/devdasrahul/autovoice-automotive-ner hf-model
cd hf-model

# 3. Copy files from AutoVoice
cp ../models/ner/* .

# 4. Track large weights with LFS
git lfs track "*.safetensors"
git add .gitattributes
git add .
git commit -m "Upload DistilBERT automotive NER model"
git push origin main
```

---

## 3. How to Deploy as a Hugging Face Space (Streamlit UI)

If you would also like to host the interactive web dashboard on Hugging Face Spaces:

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Set:
   - **Space name:** `autovoice`
   - **License:** `mit`
   - **Space SDK:** Select **Streamlit**
   - **Hardware:** `CPU basic • 2 vCPU • 16 GB • Free`
   - **Visibility:** Public
   - Click **Create Space**.
3. Clone the space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/devdasrahul/autovoice hf-space
   cd hf-space
   ```
4. Copy AutoVoice project files into `hf-space`:
   - Copy `streamlit_app.py` as `app.py`
   - Copy `requirements.txt`
   - Copy the `src/`, `models/`, and `data/` directories
5. Push to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy AutoVoice to Hugging Face Spaces"
   git push
   ```
6. Hugging Face Spaces will automatically build the container and launch your live app at `https://huggingface.co/spaces/devdasrahul/autovoice`!
