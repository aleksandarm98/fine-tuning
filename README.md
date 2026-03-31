# DMS Fine-Tuning: Mistral 7B Instruct za matematička takmičenja

Fine-tuning modela **Mistral-7B-Instruct-v0.3** za rešavanje zadataka sa takmičenja
Društva Matematičara Srbije (DMS) za srednje škole, korišćenjem **QLoRA** metode.

## Dataset

348 zadataka sa DMS takmičenja (2022–2024), podeljenih u:

| Split | N | Opis |
|-------|---|------|
| Train | 192 | Ceo 2022 (116) + 2023 bez okružnog (76) |
| Val | 39 | Okružno 2023 (srednji nivo težine) |
| Test | 116 | Ceo 2024 (hold-out, nikad se ne trenira) |

## Pokretanje

### Docker (preporučeno)

```bash
# Podesi config.yaml po potrebi, zatim:
docker compose up --build
```

Zahteva: NVIDIA GPU ≥ 16 GB VRAM, `nvidia-container-toolkit`.

### Ručno (na GPU serveru)

```bash
pip install -r requirements.txt

# 1. Priprema dataseta
python scripts/prepare_dataset.py

# 2. Fine-tuning (~30-60 min na A100)
python scripts/finetune.py

# 3. Inference (baseline + fine-tuned)
python scripts/inference.py both

# 4. Evaluacija (treba AWS Bedrock)
python scripts/evaluate.py both

# 5. Poređenje rezultata
python scripts/compare.py
```

### Lokalni test (bez GPU-a)

Postavi `dry_run: true` u `config.yaml`, zatim:

```bash
python scripts/run_pipeline.py
```

Trenira 2 koraka na 4 primera na CPU-u — samo verifikuje da pipeline ne puca.

## Metoda

- **Model**: Mistral-7B-Instruct-v0.3 (opšti model, nije specijalizovan za matematiku)
- **Fine-tuning**: QLoRA — 4-bit NF4 kvantizacija + LoRA adapteri (rank=16, α=32)
- **Evaluacija**: Multi-judge LLM sistem (Claude 3.7 Sonnet):
  - **Studija A**: Tačnost konačnog odgovora (binary 0/1)
  - **Studija B**: Logička koherentnost (0.00–1.00)
  - **Studija C**: Kvalitet objašnjenja na srpskom (0.00–1.00)
  - **Agregatni skor**: 0.30×A + 0.40×B + 0.30×C

## Struktura

```
├── config.yaml           # Svi parametri na jednom mestu
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/
│   └── raw/              # 348 originalnih zadataka (12 JSON fajlova)
├── scripts/
│   ├── config_loader.py  # Učitava config.yaml
│   ├── prepare_dataset.py
│   ├── finetune.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── compare.py
│   └── run_pipeline.py   # Orkestrator (pokreće sve redom)
└── output/               # Generisano tokom treninga (gitignored)
    ├── checkpoints/
    ├── model/
    └── results/
```

## Konfiguracija

Svi parametri su u `config.yaml`:

- `dry_run`: `true` za lokalni test bez GPU-a
- `qlora.rank`, `qlora.alpha`: LoRA hiperparametri
- `training.num_epochs`, `training.learning_rate`: trening parametri
- `evaluation.aws_profile`: AWS profil za Bedrock pristup
