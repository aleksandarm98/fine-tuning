# Pokretanje DMS Fine-Tuning Pipeline-a

## Preduslov

- Docker i Docker Compose instalirani
- Za GPU mod: NVIDIA GPU sa >=16 GB VRAM + NVIDIA Container Toolkit
- Za evaluaciju (korak 4-5): AWS CLI konfigurisan sa Bedrock pristupom

## Struktura

Pipeline ima 5 koraka:

| Korak | Opis | Zahteva |
|-------|------|---------|
| 1 | Prepare Dataset — deli 348 zadataka na train/val/test | CPU |
| 2 | Fine-Tune (QLoRA) — trenira LoRA adaptere | GPU (ili CPU za dry run) |
| 3 | Inference — generiše rešenja za test set | GPU (ili CPU za dry run) |
| 4 | Evaluate — multi-judge evaluacija via AWS Bedrock | CPU + AWS Bedrock |
| 5 | Compare — statistička analiza baseline vs fine-tuned | CPU |

Default pokretanje izvršava korake 1-3. Koraci 4-5 zahtevaju `--full` flag i AWS Bedrock pristup.

---

## 1. Konfiguracija okruženja (.env)

Kopiraj `.env.example` u `.env`:

```bash
cp .env.example .env
```

Sadržaj `.env`:

```env
# HuggingFace cache — čuva skinute modele/tokenizere između pokretanja
HF_HOME=/cache/huggingface

# AWS (potrebno samo za evaluaciju — korak 4)
# AWS_PROFILE=finetune
# AWS_DEFAULT_REGION=us-east-1
```

`HF_HOME` je obavezan — bez njega Docker skida model iznova pri svakom pokretanju (~14 GB za pravi run). Docker Compose mountuje named volume `hf_cache` na ovaj path, tako da keš preživljava rebuild-ove.

Za evaluaciju (korak 4): otkomentariši AWS varijable i postavi odgovarajući profil sa Bedrock pristupom.

---

## 2. Dry Run — verifikacija pipeline-a (bez GPU-a)

Dry run proverava da sav kod radi bez skidanja punog modela (14 GB) i bez GPU-a.

### Šta radi dry run?

- **Korak 1**: Priprema dataset normalno (train=192, val=39, test=116)
- **Korak 2**: Učitava tiny test model (~par MB umesto 14 GB), trenira 2 koraka na 4 primera
- **Korak 3**: Generiše rešenja za 3 test zadatka sa max 64 tokena

Rezultati nemaju smisla (random model), ali cilj je verifikacija da pipeline ne puca.

### Pokretanje

1. U `config.yaml` postavi:

```yaml
dry_run: true
```

2. Pokreni:

```bash
docker compose --profile cpu up --build
```

3. Očekivani output: pipeline prođe kroz sva 3 koraka bez greške, na kraju ispiše `PIPELINE COMPLETE`.

---

## 3. Pravi Fine-Tuning (GPU)

### Pokretanje na GPU mašini

1. U `config.yaml` postavi:

```yaml
dry_run: false
```

2. Pokreni:

```bash
docker compose --profile gpu up --build
```

### Šta radi pravi run?

- **Korak 1**: Priprema dataset (isto kao dry run)
- **Korak 2**: Skida Mistral-7B-Instruct-v0.3 (~14 GB), 4-bit QLoRA fine-tuning
  - 5 epoha, effective batch 16, LR 2e-4 cosine
  - Early stopping (patience=3) na validacionom loss-u
  - Čuva LoRA adaptere u `output/model/`
  - Očekivano trajanje: ~30-60 min (A100), ~1-2h (A10G)
- **Korak 3**: Inference za baseline i fine-tuned model na 116 test zadataka
  - Čuva predikcije u `output/results/predictions_{baseline,finetuned}.jsonl`

### Praćenje treninga

Logovi se ispisuju u Docker stdout. Training i validation loss se loguju na svakih 5 koraka. Training log se čuva u `output/model/training_log.json`.

---

## 4. Evaluacija i poređenje (koraci 4-5)

Zahteva AWS Bedrock pristup (Claude 3.7 Sonnet kao sudija).

### Pokretanje

Evaluacija se pokreće **nakon** koraka 1-3. Može se pokrenuti posebno:

```bash
# Samo evaluacija (korak 4)
docker compose --profile cpu run finetune-cpu python scripts/run_pipeline.py --step 4

# Samo poređenje (korak 5)
docker compose --profile cpu run finetune-cpu python scripts/run_pipeline.py --step 5
```

Ili sve odjednom (koraci 1-5):

```bash
docker compose --profile gpu run finetune-gpu python scripts/run_pipeline.py --full
```

### Output

- `output/results/evaluated_{baseline,finetuned}.jsonl` — ocene po zadatku
- `output/results/comparison_report.txt` — statistički izveštaj
- `output/results/comparison_per_task.csv` — detalji po zadatku (za Excel)

---

## 5. Pokretanje pojedinačnih koraka

Svaki korak se može pokrenuti zasebno sa `--step N`:

```bash
# Samo priprema dataseta
docker compose --profile cpu run finetune-cpu python scripts/run_pipeline.py --step 1

# Samo fine-tuning
docker compose --profile gpu run finetune-gpu python scripts/run_pipeline.py --step 2

# Samo inference
docker compose --profile gpu run finetune-gpu python scripts/run_pipeline.py --step 3
```

---

## 6. Hiperparametri

Svi hiperparametri su u `config.yaml`. Najvažniji za eksperimentisanje:

| Parametar | Default | Opis |
|-----------|---------|------|
| `qlora.rank` | 16 | LoRA rank (probati: 8, 32, 64) |
| `qlora.alpha` | 32 | LoRA alpha (obično 2x rank) |
| `training.num_epochs` | 5 | Broj epoha (probati: 3, 10) |
| `training.learning_rate` | 2e-4 | Learning rate (probati: 1e-4, 5e-4) |
| `training.batch_size` | 2 | Per-device batch size |
| `training.gradient_accumulation_steps` | 8 | Effective batch = batch_size x ovo |
| `training.early_stopping_patience` | 3 | Koliko eval koraka bez poboljšanja pre zaustavljanja |

---

## Troubleshooting

### `ValueError: ... torch to at least v2.6`
Docker image koristi PyTorch 2.6+. Ako se pojavi ova greška, rebuild-uj image: `docker compose build --no-cache`.

### `ERROR: Nema GPU-a`
Za CPU testiranje postavi `dry_run: true` u `config.yaml`. Za pravi fine-tuning je neophodan NVIDIA GPU.

### Model se skida iznova pri svakom pokretanju
Proveri da je `HF_HOME` postavljen u `.env` i da `hf_cache` volume postoji (`docker volume ls`).

### Evaluacija ne radi (korak 4)
Proveri AWS kredencijale i Bedrock pristup. U `.env` otkomentariši i postavi `AWS_PROFILE` i `AWS_DEFAULT_REGION`.
