# Training image — size and identity

| Field | Value |
| --- | --- |
| Image | `ghcr.io/nsannn/mlops-nsannn/train:sha-af2c8df` |
| Digest | `sha256:c147fe392bf29b6c712763dd396c70fa2f081f0294eaab8f91210c22d8a82e11` |
| Built from commit | `af2c8df` (Added Dockerfile: multi-stage training image) |
| Platform | linux/amd64 |
| Built | 2026-10-21 |
| Size (`docker image inspect .Size`) | 2 069 710 530 B ≈ **2 070 MB** (NFR-11 target ≤ 2 200 MB) |
| Python venv `/opt/venv` | 1 863 MB (CPU-only torch is most of it) |
| Baked model `/opt/hf` (`google/bert_uncased_L-4_H-256_A-4` @ `387825c`) | 130 MB 

Measured with Docker on WSL2. MB = 1 000 000 bytes.