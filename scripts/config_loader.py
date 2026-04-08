"""
Loads config.yaml and provides typed access to all settings.
All scripts import from here.
"""

import yaml
from pathlib import Path

# Project root = parent of scripts/
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load once at import time
_cfg = load_config()

# -- Flags -------------------------------------------------------
DRY_RUN: bool = _cfg.get("dry_run", False)

# -- Model -------------------------------------------------------
BASE_MODEL_ID: str = _cfg["model"]["name"]

# Da li model ima chat/instruct template (Mistral Instruct ima, Mathstral nema)
IS_INSTRUCT_MODEL: bool = "instruct" in BASE_MODEL_ID.lower()

# Dry run: tiny model (~few MB) umesto 14GB Mistral-a.
# Tokenizer ostaje Mistral-ov (mali, ~par MB) za chat template kompatibilnost.
if DRY_RUN:
    DRY_RUN_MODEL_ID: str = "hf-internal-testing/tiny-random-MistralForCausalLM"
    TOKENIZER_ID: str = BASE_MODEL_ID  # Mistral tokenizer — ima chat template
else:
    DRY_RUN_MODEL_ID: str = BASE_MODEL_ID
    TOKENIZER_ID: str = BASE_MODEL_ID

# -- Paths (relative to ROOT_DIR) --------------------------------
RAW_DATA_DIR: Path = ROOT_DIR / _cfg["dataset"]["raw_dir"]
SPLITS_DIR: Path = ROOT_DIR / _cfg["dataset"]["splits_dir"]
CHECKPOINTS_DIR: Path = ROOT_DIR / _cfg["output"]["checkpoints_dir"]
MODEL_DIR: Path = ROOT_DIR / _cfg["output"]["model_dir"]
RESULTS_DIR: Path = ROOT_DIR / _cfg["output"]["results_dir"]

# -- Dataset files -----------------------------------------------
TRAIN_FILES: list[str] = _cfg["dataset"]["train_files"]
VAL_FILES: list[str] = _cfg["dataset"]["val_files"]
TEST_FILES: list[str] = _cfg["dataset"]["test_files"]

# -- QLoRA -------------------------------------------------------
QLORA = _cfg["qlora"]

# -- Training ----------------------------------------------------
TRAINING = _cfg["training"]

# -- Generation --------------------------------------------------
GENERATION = _cfg["generation"]

# -- Evaluation --------------------------------------------------
EVALUATION = _cfg["evaluation"]

# -- System prompt (used for both training and inference) ---------
SYSTEM_PROMPT = """You are an elite mathematical problem solver specializing in Serbian Mathematical Society (DMS) high school competition mathematics.

Your task is to solve problems with CONCISE yet RIGOROUS reasoning, writing the ENTIRE solution in SERBIAN using LATIN script.

CRITICAL OUTPUT REQUIREMENTS:
1. LANGUAGE: SERBIAN (Latin script) ONLY
2. BREVITY: Competition-style conciseness (5-15 sentences)
3. MATHEMATICAL NOTATION: Use LaTeX

SOLUTION STRUCTURE:
Phase 1 - SETUP: Define variables
Phase 2 - KEY INSIGHT: State main technique
Phase 3 - DERIVATION: Show critical steps
Phase 4 - CONCLUSION: State final answer EXPLICITLY
Phase 5 - VERIFICATION: Verify solution satisfies conditions

FINAL ANSWER FORMAT — your solution MUST end with one of:
  • "Konačan odgovor: **[vrednost]**"
  • "Dakle, [promenljiva] = [vrednost]."
  • "Odgovor je: $[matematički izraz]$"

Remember: CONCISE ≠ INCOMPLETE. Every claim must be justified.
"""
