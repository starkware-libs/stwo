# Build and push

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

docker build .github/runners -t us-central1-docker.pkg.dev/starkware-thirdparties/github/actions-runner:latest

docker push us-central1-docker.pkg.dev/starkware-thirdparties/github/actions-runner:latest
```

