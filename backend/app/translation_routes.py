from pathlib import Path
from functools import lru_cache

import sentencepiece as spm
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_experiment"
TOKENIZER_DIR = PROJECT_ROOT / "artifacts" / "sentencepiece"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID = 0
SOS_ID = 2
EOS_ID = 3
MAX_LENGTH = 50

MODEL_VARIANTS = ("vanilla", "cifg", "wmc")

router = APIRouter(tags=["Translation"])


# ============================================================
# TOKENIZERS
# ============================================================

@lru_cache(maxsize=1)
def load_tokenizers():
    english_path = TOKENIZER_DIR / "english.model"
    darija_path = TOKENIZER_DIR / "darija.model"

    if not english_path.exists():
        raise FileNotFoundError(
            f"Tokenizer anglais introuvable : {english_path}"
        )

    if not darija_path.exists():
        raise FileNotFoundError(
            f"Tokenizer darija introuvable : {darija_path}"
        )

    english_tokenizer = spm.SentencePieceProcessor(
        model_file=str(english_path)
    )

    darija_tokenizer = spm.SentencePieceProcessor(
        model_file=str(darija_path)
    )

    return english_tokenizer, darija_tokenizer


# ============================================================
# CHARGEMENT DES TROIS MODÈLES
# ============================================================

def load_model(variant, english_tokenizer, darija_tokenizer):
    checkpoint_path = CHECKPOINT_DIR / f"{variant}_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    model_variant = checkpoint.get("variant", variant)

    cell_type = checkpoint.get(
        "cell_type",
        checkpoint.get("variant", variant),
    )

    # Évite de charger accidentellement un checkpoint
    # dans une architecture d'une autre variante.
    if cell_type != variant and model_variant != variant:
        raise ValueError(
            f"Checkpoint incohérent pour {variant} : "
            f"variant={model_variant}, cell_type={cell_type}"
        )

    model = Seq2Seq(
        source_vocab_size=checkpoint.get(
            "source_vocab_size",
            checkpoint.get(
                "src_vocab_size",
                english_tokenizer.get_piece_size(),
            ),
        ),
        target_vocab_size=checkpoint.get(
            "target_vocab_size",
            checkpoint.get(
                "tgt_vocab_size",
                darija_tokenizer.get_piece_size(),
            ),
        ),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_size=checkpoint["hidden_size"],
        cell_type=cell_type,
        pad_idx=PAD_ID,
        sos_idx=SOS_ID,
        eos_idx=EOS_ID,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Modèle chargé : {variant.upper()} ({checkpoint_path.name})")

    return model


@lru_cache(maxsize=1)
def load_all_models():
    english_tokenizer, darija_tokenizer = load_tokenizers()

    models = {}

    for variant in MODEL_VARIANTS:
        models[variant] = load_model(
            variant,
            english_tokenizer,
            darija_tokenizer,
        )

    return models, english_tokenizer, darija_tokenizer


# ============================================================
# REQUÊTE ET DÉCODAGE
# ============================================================

class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Phrase anglaise à traduire",
    )


def decode_generated(ids, tokenizer):
    clean_ids = []

    for token_id in ids:
        token_id = int(token_id)

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, SOS_ID):
            continue

        clean_ids.append(token_id)

    return tokenizer.decode(clean_ids).strip()


def translate_with_model(model, text, english_tokenizer, darija_tokenizer):
    source_ids = english_tokenizer.encode(text, out_type=int)

    if not source_ids:
        return ""

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    source_lengths = torch.tensor(
        [len(source_ids)],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():
        generated_ids = model.translate(
            source,
            source_lengths=source_lengths,
            max_length=MAX_LENGTH,
        )

    return decode_generated(
        generated_ids[0].detach().cpu().tolist(),
        darija_tokenizer,
    )


# ============================================================
# ROUTE : TRADUIRE AVEC LES TROIS MODÈLES
# ============================================================

@router.post("/translate")
def translate(request: TranslationRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Veuillez saisir une phrase en anglais.",
        )

    try:
        models, english_tokenizer, darija_tokenizer = load_all_models()

        translations = {}

        for variant, model in models.items():
            translations[variant] = translate_with_model(
                model,
                text,
                english_tokenizer,
                darija_tokenizer,
            )

        return {
            "input": text,
            "translations": translations,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Erreur pendant la traduction : {error}")

        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue pendant la traduction.",
        ) from error