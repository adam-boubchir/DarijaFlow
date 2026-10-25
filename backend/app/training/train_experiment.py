
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import torch.nn as nn
from tqdm import tqdm

from backend.app.data.translation_dataset import create_dataloader
from backend.app.models.seq2seq import Seq2Seq


ROOT_DIR = Path(__file__).resolve().parents[3]

BATCH_SIZE = 32
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0005
GRAD_CLIP = 5.0
TEACHER_FORCING_RATIO = 0.5
MAX_TRANSLATION_LENGTH = 50

PAD_IDX = 0
SOS_IDX = 2
EOS_IDX = 3

VARIANTS = ["vanilla", "wmc", "cifg"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, dataloader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
        description = "Entraînement"
    else:
        model.eval()
        description = "Validation loss"

    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(
        dataloader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in progress_bar:
        source = batch["source"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        source_lengths = batch["source_lengths"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(
                source=source,
                target=target,
                source_lengths=source_lengths,
                teacher_forcing_ratio=(
                    TEACHER_FORCING_RATIO if training else 0.0
                ),
            )

            expected = target[:, 1:]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                expected.reshape(-1),
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

        valid_tokens = expected.ne(PAD_IDX).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            moyenne=f"{total_loss / max(total_tokens, 1):.4f}",
        )

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_chrf(
    model,
    dataloader,
    darija_tokenizer,
):
    model.eval()

    hypotheses = []
    references = []

    for batch in tqdm(
        dataloader,
        desc="Calcul chrF",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        source = batch["source"].to(DEVICE)
        source_lengths = batch["source_lengths"]

        target = batch["target"]
        target_lengths = batch["target_lengths"]

        generated = model.translate(
            source=source,
            source_lengths=source_lengths,
            max_length=MAX_TRANSLATION_LENGTH,
        )

        for i in range(generated.size(0)):
            # Traduction générée : arrêter à EOS.
            predicted_ids = []

            for token_id in generated[i].tolist():
                if token_id == EOS_IDX:
                    break

                if token_id not in (PAD_IDX, SOS_IDX):
                    predicted_ids.append(token_id)

            hypothesis = darija_tokenizer.decode(predicted_ids)

            # Référence : enlever SOS, EOS et padding.
            target_length = int(target_lengths[i].item())
            reference_ids = target[i, 1:target_length - 1].tolist()

            reference = darija_tokenizer.decode(reference_ids)

            hypotheses.append(hypothesis)
            references.append(reference)

    if not hypotheses:
        return 0.0

    return sacrebleu.corpus_chrf(
        hypotheses,
        [references],
    ).score


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_chrf",
        "epoch_seconds",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_variant(
    variant,
    source_vocab_size,
    target_vocab_size,
    train_loader,
    validation_loader,
    darija_tokenizer,
):
    print("\n" + "=" * 70)
    print(f"MODÈLE : {variant.upper()}")
    print("=" * 70)

    set_seed(42)

    model = Seq2Seq(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        cell_type=variant,
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
    ).to(DEVICE)

    parameter_count = model.count_parameters()

    print(f"Appareil : {DEVICE}")
    print(f"Paramètres entraînables : {parameter_count:,}")
    print(f"Époques maximales : {EPOCHS}")
    print(f"Taille des batches : {BATCH_SIZE}")
    print(f"Taux d'apprentissage : {LEARNING_RATE}")
    print(f"Gradient clipping : {GRAD_CLIP}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    checkpoint_dir = ROOT_DIR / "checkpoints_experiment"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = (
       ROOT_DIR / "results_experiment" / "history" / f"{variant}_history.csv"
    )

    checkpoint_path = checkpoint_dir / f"{variant}_best.pt"

    history = []
    best_validation_chrf = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = 0

    epoch_bar = tqdm(
        range(1, EPOCHS + 1),
        desc=f"Époques {variant.upper()}",
        unit="époque",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        epoch_start = time.time()

        print(f"\n[{variant.upper()}] Époque {epoch}/{EPOCHS}")

        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        validation_loss = run_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            optimizer=None,
        )

        validation_chrf = evaluate_chrf(
            model=model,
            dataloader=validation_loader,
            darija_tokenizer=darija_tokenizer,
        )

        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(validation_loss, 6),
            "validation_chrf": round(validation_chrf, 4),
            "epoch_seconds": round(elapsed, 2),
        }

        history.append(row)
        save_history(history, history_path)

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{validation_loss:.4f}",
            chrF=f"{validation_chrf:.2f}",
            temps=f"{elapsed:.1f}s",
        )

        print(
            f"Train loss : {train_loss:.4f} | "
            f"Validation loss : {validation_loss:.4f} | "
            f"Validation chrF : {validation_chrf:.2f} | "
            f"Durée : {elapsed:.1f}s"
        )

        # Le meilleur modèle est sélectionné par chrF de validation.
        if (
            np.isfinite(validation_chrf)
            and validation_chrf > best_validation_chrf
        ):
            best_validation_chrf = validation_chrf
            best_validation_loss = validation_loss
            best_epoch = epoch

            torch.save(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_chrf": validation_chrf,
                    "train_loss": train_loss,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_size": HIDDEN_SIZE,
                    "source_vocab_size": source_vocab_size,
                    "target_vocab_size": target_vocab_size,
                    "parameter_count": parameter_count,
                    "tokenizer_type": "sentencepiece_bpe",
                },
                checkpoint_path,
            )

            print(
                f"Meilleur checkpoint sauvegardé "
                f"(époque {epoch}, chrF {validation_chrf:.2f})"
            )

    result = {
        "variant": variant,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss if best_epoch else None
        ),
        "best_validation_chrf": (
            best_validation_chrf if best_epoch else None
        ),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }

    print(f"\nFin du modèle {variant.upper()}")
    print(f"Meilleure époque : {best_epoch}")
    print(f"Meilleur chrF validation : {best_validation_chrf:.2f}")
    print(f"Checkpoint : {checkpoint_path}")

    return result


def main():
    set_seed(42)

    print("=" * 70)
    print("TRADUCTION ANGLAIS -> DARIJA")
    print("SentencePiece BPE + Bahdanau Attention")
    print("Comparaison Vanilla / WMC / CIFG")
    print("=" * 70)

    print(f"Appareil : {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    print("\nChargement des données...")

    train_loader, english_tokenizer, darija_tokenizer = (
        create_dataloader(
            "train",
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    validation_loader, _, _ = create_dataloader(
        "validation",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    source_vocab_size = english_tokenizer.vocab_size()
    target_vocab_size = darija_tokenizer.vocab_size()

    print(f"Vocabulaire anglais : {source_vocab_size}")
    print(f"Vocabulaire Darija : {target_vocab_size}")
    print(f"Batches entraînement : {len(train_loader)}")
    print(f"Batches validation : {len(validation_loader)}")

    results = []

    for variant in VARIANTS:
        result = train_variant(
            variant=variant,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            train_loader=train_loader,
            validation_loader=validation_loader,
            darija_tokenizer=darija_tokenizer,
        )
        results.append(result)

    summary_path = (
        ROOT_DIR / "results_experiment" / "metrics" / "training_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("ENTRAÎNEMENTS TERMINÉS")
    print("=" * 70)

    for result in results:
        print(
            f"{result['variant'].upper()} | "
            f"meilleure époque : {result['best_epoch']} | "
            f"validation chrF : {result['best_validation_chrf']}"
        )

    print(f"\nRésumé : {summary_path}")


if __name__ == "__main__":
    main()
