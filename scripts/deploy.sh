#!/usr/bin/env bash
# ==============================================================================
# AutoVoice Production Deployment Script (GCP Cloud Run - Always Free Tier)
# ==============================================================================
# Constraints enforced:
# - Region: us-central1 (eligible for GCP Always Free tier)
# - CPU: 1 vCPU, Memory: 1GiB
# - Min instances: 0 (scales to zero on idle, zero idle compute cost)
# - Max instances: 2 (hard ceiling against runaway requests/charges)
# - Artifact Registry retention: retains last 3 image tags to fit in 0.5GB quota
# ==============================================================================

set -euo pipefail

REGION="us-central1"
REPO_NAME="autovoice-repo"
API_SERVICE_NAME="autovoice-api"
UI_SERVICE_NAME="autovoice-ui"

# Discover GCP project
PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null || true)}"
if [ -z "${PROJECT_ID}" ]; then
  echo "Error: No GCP Project ID provided or configured."
  echo "Usage: ./scripts/deploy.sh <GCP_PROJECT_ID>"
  exit 1
fi

echo "=========================================================="
echo " Deploying AutoVoice to GCP Cloud Run (Always Free Tier)"
echo " Project: ${PROJECT_ID}"
echo " Region:  ${REGION}"
echo "=========================================================="

# 1. Enable required GCP services (idempotent)
echo "[1/6] Enabling necessary GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  bigquery.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Create Artifact Registry repository if not present
echo "[2/6] Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Docker repository for AutoVoice microservices" \
    --project="${PROJECT_ID}"
fi

IMAGE_PREFIX="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
API_IMAGE="${IMAGE_PREFIX}/${API_SERVICE_NAME}:$(date +%Y%m%d%H%M%S)"
UI_IMAGE="${IMAGE_PREFIX}/${UI_SERVICE_NAME}:$(date +%Y%m%d%H%M%S)"

# 3. Build container images via Cloud Build (120 free build-min/day)
echo "[3/6] Building API image with Cloud Build..."
gcloud builds submit --tag "${API_IMAGE}" --file Dockerfile . --project="${PROJECT_ID}"

echo "[4/6] Building UI image with Cloud Build..."
gcloud builds submit --tag "${UI_IMAGE}" --file Dockerfile.ui . --project="${PROJECT_ID}"

# 4. Deploy API service to Cloud Run
echo "[5/6] Deploying API Service to Cloud Run..."
gcloud run deploy "${API_SERVICE_NAME}" \
  --image="${API_IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=2 \
  --cpu=1 \
  --memory=1Gi \
  --timeout=60 \
  --set-env-vars="ENVIRONMENT=prod,LOG_LEVEL=INFO" \
  --project="${PROJECT_ID}"

API_URL=$(gcloud run services describe "${API_SERVICE_NAME}" --region="${REGION}" --format="value(status.url)" --project="${PROJECT_ID}")
echo "API deployed at: ${API_URL}"

# 5. Deploy UI service to Cloud Run
echo "[6/6] Deploying Streamlit UI Service to Cloud Run..."
gcloud run deploy "${UI_SERVICE_NAME}" \
  --image="${UI_IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=2 \
  --cpu=1 \
  --memory=1Gi \
  --timeout=120 \
  --set-env-vars="AUTOVOICE_API_URL=${API_URL}" \
  --project="${PROJECT_ID}"

UI_URL=$(gcloud run services describe "${UI_SERVICE_NAME}" --region="${REGION}" --format="value(status.url)" --project="${PROJECT_ID}")

# 6. Artifact Registry cleanup: Keep only the latest 3 images to respect free 0.5 GB quota
echo "Cleaning older container images in Artifact Registry (retaining 3 latest)..."
for SERVICE in "${API_SERVICE_NAME}" "${UI_SERVICE_NAME}"; do
  IMAGES_TO_DELETE=$(gcloud artifacts docker images list "${IMAGE_PREFIX}/${SERVICE}" \
    --sort-by=~CREATE_TIME --format="value(DIGEST)" | tail -n +4 || true)
  if [ -n "${IMAGES_TO_DELETE}" ]; then
    for DIGEST in ${IMAGES_TO_DELETE}; do
      gcloud artifacts docker images delete "${IMAGE_PREFIX}/${SERVICE}@${DIGEST}" --quiet || true
    done
  fi
done

echo "=========================================================="
echo " AutoVoice Successfully Deployed!"
echo " - API Endpoint:  ${API_URL}/docs"
echo " - Streamlit UI:  ${UI_URL}"
echo " - Monthly Cost:  \$0.00 (within Always Free limits)"
echo "=========================================================="
