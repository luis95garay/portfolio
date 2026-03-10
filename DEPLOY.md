# Deploy Portfolio to Google Cloud Run

CI/CD via GitHub Actions: cada push a `main` construye la imagen Docker,
la sube a Artifact Registry y despliega automáticamente en Cloud Run.

---

## Resumen del flujo

```
git push → main
    └── GitHub Actions
          ├── docker build + push → Artifact Registry
          └── gcloud run deploy  → Cloud Run  ✅
```

---

## Parte 1 — Setup inicial en GCP (solo una vez)

### 1.1 Prerrequisitos

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) instalado
- Proyecto GCP creado con billing habilitado

### 1.2 Configura el proyecto y habilita APIs

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com
```

### 1.3 Crea el repositorio en Artifact Registry

```bash
gcloud artifacts repositories create portfolio \
  --repository-format=docker \
  --location=us-central1 \
  --description="Portfolio static site"
```

### 1.4 Crea un Service Account para GitHub Actions

```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions — Portfolio Deploy"
```

### 1.5 Asigna los permisos necesarios

```bash
PROJECT_ID=YOUR_PROJECT_ID
SA=github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com

# Subir imágenes a Artifact Registry
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/artifactregistry.writer"

# Desplegar en Cloud Run
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/run.admin"

# Actuar como service account en el deploy
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/iam.serviceAccountUser"
```

### 1.6 Genera la clave JSON del Service Account

```bash
gcloud iam service-accounts keys create key.json \
  --iam-account=$SA
```

> **Importante:** `key.json` contiene credenciales sensibles.
> No lo subas al repositorio — está en `.gitignore` automáticamente si agregas `key.json`.

---

## Parte 2 — Configura los Secrets en GitHub

Ve a tu repositorio en GitHub:
**Settings → Secrets and variables → Actions → New repository secret**

| Secret | Valor |
|--------|-------|
| `GCP_PROJECT_ID` | Tu Project ID de GCP (ej. `my-portfolio-123456`) |
| `GCP_SA_KEY` | El contenido completo del archivo `key.json` (copia y pega el JSON) |

Para copiar el contenido del key en la terminal:

```bash
# macOS / Linux
cat key.json | pbcopy       # macOS
cat key.json | xclip        # Linux

# Windows (PowerShell)
Get-Content key.json | Set-Clipboard
```

Luego elimina el archivo local:

```bash
rm key.json
```

---

## Parte 3 — Primer deploy

Con los secrets configurados, haz push a `main`:

```bash
git add .
git commit -m "Setup CI/CD"
git push origin main
```

GitHub Actions se activará automáticamente. Puedes ver el progreso en:
**Actions → Deploy to Cloud Run**

Al finalizar, el último paso imprime la URL pública del servicio.

---

## Parte 4 — Deploys posteriores (flujo normal)

A partir de ahora, cualquier cambio a `main` se despliega solo:

```bash
# Edita index.html, agrega imágenes, etc.
git add .
git commit -m "Update portfolio"
git push origin main
# → GitHub Actions buildea y despliega automáticamente
```

---

## Deploy manual (emergencia)

Si necesitas desplegar sin pasar por GitHub Actions:

```bash
IMAGE=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/portfolio/site

gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -t $IMAGE:latest .
docker push $IMAGE:latest

gcloud run deploy portfolio \
  --image $IMAGE:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

---

## Dominio personalizado (opcional)

1. Ve a **Cloud Run → portfolio → Manage Custom Domains**
2. Agrega tu dominio y sigue los pasos de verificación DNS

---

## Archivos del proyecto

| Archivo | Propósito |
|---------|-----------|
| `Dockerfile` | Imagen nginx:alpine con los archivos del sitio |
| `nginx.conf` | Sirve en puerto 8080, gzip, cache de imágenes |
| `.dockerignore` | Excluye archivos innecesarios del build |
| `.github/workflows/deploy.yml` | Pipeline CI/CD — se activa en cada push a `main` |
