# Pokretanje DMS Fine-Tuning Pipeline-a

## Preduslov

- Docker i Docker Compose
- Za trening: NVIDIA GPU ≥ 16 GB VRAM + NVIDIA Container Toolkit
- Za evaluaciju: AWS CLI sa Bedrock pristupom

## Quick Start

```bash
# 1. Kreiraj .env
cp .env.example .env

# 2. Izaberi konfiguraciju (kopiraj u config.yaml)
cp configs/iter8a_attention_rank16.yaml config.yaml

# 3. Dry run (provera da pipeline radi)
#    Prvo postavi dry_run: true u config.yaml
docker compose --profile dry-run up --build

# 4. Pravi trening na GPU serveru
#    Postavi dry_run: false u config.yaml
docker compose --profile train up --build
```

---

## Izbor konfiguracije

Postoje 4 varijante. Kopiraj željenu u `config.yaml`:

```bash
# Varijanta A — rank 16, samo attention moduli
cp configs/iter8a_attention_rank16.yaml config.yaml

# Varijanta B — isti kao iter 7, agresivniji early stopping
cp configs/iter8b_early_stop.yaml config.yaml

# Varijanta C — konzervativna, rank 8, samo q/v
cp configs/iter8c_conservative.yaml config.yaml

# Varijanta D — Mathstral-7B (math-specijalizovan base model)
cp configs/iter9_mathstral.yaml config.yaml
```

| Varijanta | Model | rank | Moduli | LR | Opis |
|:---------:|:-----:|:----:|:------:|:--:|:-----|
| A | Mistral Instruct | 16 | q/k/v/o | 2e-4 | Pun kapacitet, zaštićen MLP |
| B | Mistral Instruct | 16 | svih 7 | 2e-4 | Isti kao iter 7, ranije zaustavljanje |
| C | Mistral Instruct | 8 | q/v | 1e-4 | Minimalan fine-tuning |
| D | **Mathstral** | 8 | q/v | 5e-5 | Math base model, konzervativni QLoRA |

**Napomena**: Varijanta D koristi Mathstral-7B-v0.1 koji NIJE instruct model. Pipeline automatski detektuje tip modela i koristi odgovarajući prompt format (chat template za Instruct, raw prompt za base modele).

Kontejner čita `config.yaml` kroz volume mount — ne treba rebuild.

---

## Dry Run — provera pipeline-a

Dry run koristi tiny model (~par MB) na CPU-u. Proverava da kod ne puca pre slanja na GPU server.

1. Postavi `dry_run: true` u `config.yaml`
2. Pokreni:
```bash
docker compose --profile dry-run up --build
```
3. Očekivani output: pipeline prođe kroz prepare → finetune (2 koraka) → inference (3 zadatka) i ispiše `PIPELINE COMPLETE`

---

## Trening na GPU serveru

1. Postavi `dry_run: false` u `config.yaml`
2. Pokreni:
```bash
docker compose --profile train up --build
```

### Šta pipeline radi

| Korak | Opis | Trajanje |
|:-----:|:-----|:---------|
| 1 | Priprema dataseta (train/val/test split) | ~1 sec |
| 2 | QLoRA fine-tuning | ~30-60 min (A100) |
| 3 | Inference (baseline + fine-tuned, 116 zadataka) | ~30-60 min |

Rezultati se čuvaju u `output/results/predictions_{baseline,finetuned}.jsonl`.

### Pokretanje više varijanti

Između svake varijante sačuvaj rezultate i obriši output:

```bash
# 1. Pokreni varijantu A
cp configs/iter8a_attention_rank16.yaml config.yaml
docker compose --profile train up --build

# 2. Sačuvaj rezultate
cp -r output/results output/results_iter8a

# 3. Obriši output
rm -rf output/

# 4. Pokreni varijantu B
cp configs/iter8b_early_stop.yaml config.yaml
docker compose --profile train up --build

# 5. Sačuvaj rezultate
cp -r output/results output/results_iter8b

# 6. Obriši output i pokreni varijantu C
rm -rf output/
cp configs/iter8c_conservative.yaml config.yaml
docker compose --profile train up --build
cp -r output/results output/results_iter8c
```

---

## Evaluacija (posle treninga)

Evaluacija se radi lokalno sa predikcijama skinutim sa servera. Koristi Claude 3.7 Sonnet kao LLM sudiju preko AWS Bedrock.

```bash
# Kopiraj predikcije u evaluacioni direktorijum
cp output/results/predictions_baseline.jsonl /path/to/finetuned_results/
cp output/results/predictions_finetuned.jsonl /path/to/finetuned_results/

# Pokreni evaluaciju (696 API poziva, ~20 min)
cd /path/to/finetuned_results/
python run_evaluation.py both

# Generiši izveštaj
python generate_report.py
```

---

## Troubleshooting

### `bitsandbytes` greška na Mac-u
Koristi `--profile dry-run` (koristi `requirements-cpu.txt` bez bitsandbytes). Profil `train` je samo za GPU server.

### Model se skida iznova pri svakom pokretanju
Proveri da je `HF_HOME=/cache/huggingface` u `.env` i da `hf_cache` volume postoji (`docker volume ls`).

### `ERROR: Nema GPU-a`
Za lokalno testiranje koristi `dry_run: true` + `--profile dry-run`. Pravi trening zahteva NVIDIA GPU.

### Platform warning (arm64 vs amd64)
Na Mac-u `--profile dry-run` koristi `python:3.11-slim` (kompatibilan sa arm64). Profil `train` koristi CUDA image koji radi samo na x86 GPU serveru.
