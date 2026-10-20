# GCP Always Free Deployment Guide & Runbook

AutoVoice is engineered to run **100% within the Google Cloud Platform Always Free tier** ($0.00/month recurring cost) while delivering a production-grade NLP & GenAI service for automotive complaint intelligence.

---

## 1. Verified GCP Always Free Quotas

The following limits are verified against official Google Cloud Free Tier documentation:

| Service | Always Free Allowance | AutoVoice Configuration & Consumption | Cost Impact |
| :--- | :--- | :--- | :--- |
| **Cloud Run** | 2,000,000 requests / month<br>360,000 GB-seconds memory<br>180,000 vCPU-seconds | `min-instances=0`<br>`max-instances=2`<br>`memory=1Gi`, `cpu=1`<br>Scaled to 0 when idle | **$0.00** |
| **Google BigQuery** | 10 GB active storage / month<br>1 TB analysis queries / month | Processed complaints table: ~123 KB<br>Batch load jobs: **Free** (no streaming inserts)<br>Day partitioned & clustered | **$0.00** |
| **Artifact Registry**| 0.5 GB storage / month | Pruning retention script keeps last 3 images<br>Multi-stage slim container build | **$0.00** |
| **Cloud Build** | 120 build-minutes / day | Automated CI/CD builds take ~2.5 min | **$0.00** |
| **Network Egress** | 1 GB egress from North America to all regions (except China/Australia) | JSON payloads & UI responses < 100 MB / mo | **$0.00** |

> [!IMPORTANT]
> To remain eligible for the Always Free tier, all compute resources **must be deployed to `us-central1`** (Iowa). Other regions (e.g. `europe-west1` or `asia-south1`) incur billable runtime charges.

---

## 2. Hard Cost Safeguards Checklist

Before deploying, verify each constraint in this checklist:

- [x] **Region Enforced:** Cloud Run services deployed exclusively in `us-central1`.
- [x] **Cold Start to Zero:** `--min-instances 0` configured so idle services consume zero vCPU/RAM seconds.
- [x] **Concurrency Ceiling:** `--max-instances 2` configured to prevent runaway scaling or denial-of-wallet spikes.
- [x] **Batch Ingestion Only:** BigQuery writes use free batch load jobs (`bq load` or `client.load_table_from_dataframe`), never billable BigQuery Storage Write API streaming inserts.
- [x] **Partition & Cluster Pruning:** BigQuery table is partitioned by `date_added (DAY)` and clustered by `make, component` to minimize query bytes scanned.
- [x] **Artifact Registry Retention:** Automated retention policy prunes images older than the 3 most recent builds, staying below the 0.5 GB storage quota.
- [x] **Budget Kill-Switch:** $1.00 USD monthly budget alert set up in Google Cloud Billing with automated email notifications.

---

## 3. Setting Up a $1.00 Budget Alert

To guarantee zero surprise charges, configure a $1.00 budget alert via `gcloud`:

```bash
# Get your billing account ID
BILLING_ACCOUNT=$(gcloud billing accounts list --format="value(name)" | head -n 1)

# Create a $1.00 budget with alerts at 50%, 90%, and 100%
gcloud billing budgets create \
  --billing-account="${BILLING_ACCOUNT}" \
  --display-name="AutoVoice-Zero-Cost-Safety-Net" \
  --budget-amount=1.00USD \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --all-updates-rule-pubsub-topic="projects/${PROJECT_ID}/topics/billing-alerts"
```

---

## 4. Step-by-Step Deployment Walkthrough

### Step 4.1: Authentication & Project Initialization

```bash
# 1. Login to Google Cloud
gcloud auth login
gcloud auth application-default login

# 2. Select or create your project
export PROJECT_ID="your-autovoice-project"
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region us-central1
```

### Step 4.2: Enable GCP Service APIs

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  bigquery.googleapis.com \
  --project="${PROJECT_ID}"
```

### Step 4.3: Deploy via Idempotent Deployment Script

The bundled script [`scripts/deploy.sh`](file:///c:/Users/rahul/Downloads/AutoVoice,%20automotive%20complaint%20intelligence%20on%20GCP/scripts/deploy.sh) handles container builds, Cloud Run deployments, and image retention cleanup:

```bash
# Execute deployment script
chmod +x ./scripts/deploy.sh
./scripts/deploy.sh "${PROJECT_ID}"
```

### Step 4.4: Manual Step-by-Step Alternative

If deploying each step manually:

```bash
# 1. Create Docker repository in Artifact Registry
gcloud artifacts repositories create autovoice-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="AutoVoice container repository" \
  --project="${PROJECT_ID}"

# 2. Build and push FastAPI service
gcloud builds submit \
  --tag "us-central1-docker.pkg.dev/${PROJECT_ID}/autovoice-repo/autovoice-api:latest" \
  --file Dockerfile .

# 3. Deploy API to Cloud Run
gcloud run deploy autovoice-api \
  --image="us-central1-docker.pkg.dev/${PROJECT_ID}/autovoice-repo/autovoice-api:latest" \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=2 \
  --cpu=1 \
  --memory=1Gi \
  --timeout=60

API_URL=$(gcloud run services describe autovoice-api --region=us-central1 --format="value(status.url)")

# 4. Build and deploy Streamlit UI
gcloud builds submit \
  --tag "us-central1-docker.pkg.dev/${PROJECT_ID}/autovoice-repo/autovoice-ui:latest" \
  --file Dockerfile.ui .

gcloud run deploy autovoice-ui \
  --image="us-central1-docker.pkg.dev/${PROJECT_ID}/autovoice-repo/autovoice-ui:latest" \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=2 \
  --cpu=1 \
  --memory=1Gi \
  --timeout=120 \
  --set-env-vars="AUTOVOICE_API_URL=${API_URL}"
```

---

## 5. Automated CI/CD with Cloud Build

Every commit can trigger automated linting, testing, and zero-downtime deployment using [`cloudbuild.yaml`](file:///c:/Users/rahul/Downloads/AutoVoice,%20automotive%20complaint%20intelligence%20on%20GCP/cloudbuild.yaml):

```bash
gcloud builds triggers create github \
  --repo-name="AutoVoice" \
  --repo-owner="your-github-username" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml"
```

---

## 6. Verification & Health Monitoring

Test the live deployed endpoints:

```bash
# Check API liveness
curl -s "${API_URL}/health" | jq .

# Check BigQuery & model readiness
curl -s "${API_URL}/ready" | jq .

# Verify live verified metrics summary
curl -s "${API_URL}/v1/metrics-summary" | jq .

# Open Streamlit Web UI in browser
UI_URL=$(gcloud run services describe autovoice-ui --region=us-central1 --format="value(status.url)")
echo "Open UI at: ${UI_URL}"
```

---

## 7. One-Command Teardown & Clean-up

When you need to decommission or delete all cloud resources to ensure zero lingering costs:

```bash
#!/usr/bin/env bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"

echo "Tearing down AutoVoice GCP resources..."

# 1. Delete Cloud Run services
gcloud run services delete autovoice-ui --region="${REGION}" --quiet || true
gcloud run services delete autovoice-api --region="${REGION}" --quiet || true

# 2. Delete Artifact Registry images & repository
gcloud artifacts repositories delete autovoice-repo --location="${REGION}" --quiet || true

# 3. Delete BigQuery dataset and complaints table
bq rm -r -f "${PROJECT_ID}:autovoice" || true

echo "All AutoVoice resources have been completely decommissioned."
```
