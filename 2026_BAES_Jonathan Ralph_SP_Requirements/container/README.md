# Hypertension Risk Assessment - Container Deployment

This folder contains container-related assets. The actively maintained app image build is based on `src/Dockerfile`.

## Prerequisites

- Docker Desktop installed and running.

## Recommended Build and Run

Run the following commands from the repository root:

```bash
# Build using the maintained Dockerfile in src/
docker build -f src/Dockerfile -t hypertension-app src

# Run the Streamlit app container
docker run --rm -p 8501:8501 --name hypertension-instance hypertension-app
```

Then open:

```text
http://localhost:8501
```

## Common Commands

```bash
# Stop the running container
docker stop hypertension-instance

# Remove the local image
docker rmi hypertension-app
```

## Notes

- The app entrypoint is `thesis_webapp/app.py` inside the image.
- Python dependencies are installed from `src/requirements.txt` during build.
- For full local development instructions (non-container and training workflow), see `manual/DEVELOPMENT.md` and the root `README.md`.