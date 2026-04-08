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

### 1. Dry run — provera pipeline-a (lokalno, bez GPU)

Postavi `dry_run: true` u `config.yaml`, zatim:

```bash
docker compose --profile dry-run up --build
```

Trenira 2 koraka na 4 primera sa tiny modelom na CPU-u. Proverava da pipeline ne puca pre slanja na GPU server.

### 2. Trening na GPU serveru

#### Izbor konfiguracije

Kopiraj željeni config u `config.yaml` (obavezno `dry_run: false`):

```bash
# Varijanta A — rank 16, samo attention moduli (PREPORUČENO)
cp configs/iter8a_attention_rank16.yaml config.yaml

# Varijanta B — isti kao iter 7, agresivniji early stopping
cp configs/iter8b_early_stop.yaml config.yaml

# Varijanta C — konzervativna, rank 8, samo q/v
cp configs/iter8c_conservative.yaml config.yaml
```

#### Pokretanje

```bash
docker compose --profile train up --build
```

Pipeline automatski pokreće: prepare_dataset → finetune → inference (baseline + finetuned).

#### Između varijanti

Sačuvaj rezultate pa obriši output pre sledeće varijante:

```bash
# Sačuvaj rezultate (npr. za varijantu A)
cp -r output/results output/results_iter8a

# Obriši output pre sledeće varijante
rm -rf output/

# Kopiraj sledeći config i pokreni ponovo
cp configs/iter8b_early_stop.yaml config.yaml
docker compose --profile train up --build
```

Zahteva: NVIDIA GPU ≥ 16 GB VRAM, `nvidia-container-toolkit`.

## Konfiguracije

Sve konfiguracije su u `configs/` direktorijumu. Aktivna je uvek `config.yaml`.

| Config fajl | Model | Opis | rank | Moduli | LR |
|:------------|:------|:-----|:----:|:------:|:--:|
| `iter8a_attention_rank16.yaml` | Mistral Instruct | Pun kapacitet, zaštićen MLP | 16 | q/k/v/o | 2e-4 |
| `iter8b_early_stop.yaml` | Mistral Instruct | Isti kao iter 7, ranije zaustavljanje | 16 | svih 7 | 2e-4 |
| `iter8c_conservative.yaml` | Mistral Instruct | Minimalan fine-tuning | 8 | q/v | 1e-4 |
| `iter9_mathstral.yaml` | **Mathstral-7B** | Math base model, konzervativni QLoRA | 8 | q/v | 5e-5 |

### Prethodne iteracije (referenca)

| Iter | rank | Moduli | LR | Eval Loss | Ishod |
|:----:|:----:|:------:|:--:|:---------:|:------|
| 1 | 16 | svih 7 | 2e-4 | 0.951 | Overfitting od epohe 1.67 |
| 3 | 8 | q/k/v/o | 5e-5 | loš | Underfit |
| 4 | 8 | q/k/v/o | 1e-4 | bolji | I dalje underfit |
| 5/7 | 16 | svih 7 | 2e-4 | **0.930** | Najbolji train, ali lošiji od baseline-a na testu |

## Metoda

- **Model**: Mistral-7B-Instruct-v0.3
- **Fine-tuning**: QLoRA — 4-bit NF4 kvantizacija + LoRA adapteri
- **Evaluacija**: Multi-judge LLM sistem (Claude 3.7 Sonnet via AWS Bedrock):
  - **Studija A**: Matematička korektnost (0.00–1.00) — 5 dimenzija: razumevanje problema, pristup, tačnost tvrdnji, korektnost izvođenja, konačan odgovor
  - **Studija B**: Logička koherentnost (0.00–1.00)
  - **Studija C**: Kvalitet objašnjenja na srpskom (0.00–1.00)
  - **Agregatni skor**: 0.30×A + 0.40×B + 0.30×C

## Struktura

```
├── config.yaml                          # Aktivna konfiguracija (kopija iz configs/)
├── configs/
│   ├── iter8a_attention_rank16.yaml     # rank 16, attention only
│   ├── iter8b_early_stop.yaml           # iter 7 + agresivniji early stop
│   └── iter8c_conservative.yaml         # rank 8, q/v, lr 1e-4
├── Dockerfile                           # Multi-stage: python:3.11-slim (dry-run) / CUDA (train)
├── docker-compose.yml                   # Profili: dry-run, train
├── requirements.txt                     # GPU dependencies (sa bitsandbytes)
├── requirements-cpu.txt                 # CPU dependencies (bez bitsandbytes)
├── data/
│   └── raw/                             # 348 originalnih zadataka (12 JSON fajlova)
├── scripts/
│   ├── config_loader.py                 # Učitava config.yaml
│   ├── prepare_dataset.py
│   ├── finetune.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── compare.py
│   └── run_pipeline.py                  # Orkestrator (pokreće sve redom)
└── output/                              # Generisano tokom treninga (gitignored)
    ├── checkpoints/
    ├── model/
    └── results/
```
