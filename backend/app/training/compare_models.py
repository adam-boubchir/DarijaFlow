
# backend/app/training/compare_models.py

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sacrebleu.metrics import BLEU, CHRF

from backend.app.data.translation_dataset import create_dataloader
from backend.app.data.vocabulary import Vocabulary
from backend.app.models.seq2seq import Seq2Seq


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

BATCH_SIZE = 32
EMBEDDING_DIM = 64
HIDDEN_SIZE = 128
MAX_TRANSLATION_LENGTH = 50

PAD_IDX = 0
SOS_IDX = 2
EOS_IDX = 3

VARIANTS = ["vanilla", "wmc", "cifg"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES_PER_MODEL = 20


# ============================================================
# VOCABULAIRE
# ============================================================

def load_vocab(path: Path) -> Vocabulary:
    return Vocabulary.load(path)


def decode_tokens(vocab: Vocabulary, ids) -> str:
    """Convertit les identifiants en texte et enlève les tokens spéciaux."""
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()

    tokens = []

    for idx in ids:
        token = vocab.itos[idx] if 0 <= idx < len(vocab.itos) else "<unk>"

        if token == EOS_IDX:
            break

        if token in {"<pad>", "<sos>", "<eos>"}:
            continue

        tokens.append(token)

    # La tokenisation conserve la ponctuation : on la recolle proprement.
    text = " ".join(tokens)
    for punctuation in [".", ",", "!", "?", ";", ":", ")", "]", "}"]:
        text = text.replace(" " + punctuation, punctuation)

    for punctuation in ["(", "[", "{"]:
        text = text.replace(punctuation + " ", punctuation)

    return text.strip()


# ============================================================
# CHARGEMENT DES CHECKPOINTS
# ============================================================

def load_model(variant, source_vocab_size, target_vocab_size):
    checkpoint_path = CHECKPOINT_DIR / f"{variant}_best.pt"

    if not checkpoint_path.exists():
        print(f"Checkpoint introuvable : {checkpoint_path}")
        return None

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

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate_model(model, test_loader, english_vocab, darija_vocab):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    bleu_metric = BLEU()
    chrf_metric = CHRF()

    total_loss = 0.0
    total_batches = 0
    references = []
    predictions = []
    examples = []
    total_inference_seconds = 0.0
    total_sentences = 0

    with torch.no_grad():
        for batch in test_loader:
            source = batch["source"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            source_lengths = batch["source_lengths"]

            # Loss avec les vraies séquences cibles (teacher forcing).
            logits, expected = model(source, source_lengths, target)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                expected.reshape(-1),
            )

            total_loss += loss.item()
            total_batches += 1

            # Génération libre des traductions.
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            generated = model.translate(
                source,
                source_lengths,
                max_length=MAX_TRANSLATION_LENGTH,
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            total_inference_seconds += time.perf_counter() - start

            for i in range(source.size(0)):
                source_text = decode_tokens(
                    english_vocab, source[i]
                )
                reference_text = decode_tokens(
                    darija_vocab, target[i]
                )
                prediction_text = decode_tokens(
                    darija_vocab, generated[i]
                )

                references.append(reference_text)
                predictions.append(prediction_text)
                total_sentences += 1

                if len(examples) < EXAMPLES_PER_MODEL:
                    examples.append({
                        "english": source_text,
                        "reference_darija": reference_text,
                        "predicted_darija": prediction_text,
                    })

    # SacreBLEU attend une liste de prédictions et une liste
    # contenant une liste de références.
    bleu_score = bleu_metric.corpus_score(
        predictions, [references]
    ).score

    chrf_score = chrf_metric.corpus_score(
        predictions, [references]
    ).score

    return {
        "test_loss": total_loss / max(total_batches, 1),
        "bleu": bleu_score,
        "chrf": chrf_score,
        "test_examples": total_sentences,
        "total_inference_seconds": total_inference_seconds,
        "average_inference_seconds_per_sentence": (
            total_inference_seconds / max(total_sentences, 1)
        ),
        "parameters": sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        ),
        "examples": examples,
    }


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():
    print("=" * 70)
    print("ÉVALUATION DES MODÈLES ANGLAIS -> DARIJA")
    print("=" * 70)
    print(f"Appareil : {DEVICE}")

    test_loader, english_vocab, darija_vocab = create_dataloader(
        "test",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Vocabulaire anglais : {len(english_vocab)}")
    print(f"Vocabulaire darija : {len(darija_vocab)}")
    print(f"Batches de test : {len(test_loader)}")

    all_results = []

    for variant in VARIANTS:
        print(f"\nÉvaluation de {variant.upper()}...")

        model = load_model(
            variant,
            len(english_vocab),
            len(darija_vocab),
        )

        if model is None:
            continue

        metrics = evaluate_model(
            model,
            test_loader,
            english_vocab,
            darija_vocab,
        )

        result = {
            "variant": variant,
            "test_loss": metrics["test_loss"],
            "bleu": metrics["bleu"],
            "chrf": metrics["chrf"],
            "parameters": metrics["parameters"],
            "test_examples": metrics["test_examples"],
            "total_inference_seconds": metrics["total_inference_seconds"],
            "average_inference_seconds_per_sentence":
                metrics["average_inference_seconds_per_sentence"],
        }
        all_results.append(result)

        print(f"Loss test : {metrics['test_loss']:.4f}")
        print(f"BLEU      : {metrics['bleu']:.2f}")
        print(f"chrF      : {metrics['chrf']:.2f}")
        print(f"Paramètres: {metrics['parameters']:,}")
        print(
            "Temps moyen par phrase : "
            f"{metrics['average_inference_seconds_per_sentence']:.6f} s"
        )

        # Enregistrer des exemples pour inspection manuelle.
        examples_path = RESULTS_DIR / f"{variant}_translation_examples.csv"
        with examples_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "english",
                    "reference_darija",
                    "predicted_darija",
                ],
            )
            writer.writeheader()
            writer.writerows(metrics["examples"])

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Sauvegarder les métriques comparatives.
    json_path = METRICS_DIR / "model_comparison.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4, ensure_ascii=False)

    csv_path = METRICS_DIR / "model_comparison.csv"
    if all_results:
        with csv_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "variant",
                    "test_loss",
                    "bleu",
                    "chrf",
                    "parameters",
                    "test_examples",
                    "total_inference_seconds",
                    "average_inference_seconds_per_sentence",
                ],
            )
            writer.writeheader()
            writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("RÉSUMÉ COMPARATIF")
    print("=" * 70)
    print(
        f"{'Modèle':<12}{'Loss':>10}{'BLEU':>10}"
        f"{'chrF':>10}{'Paramètres':>16}"
    )

    for result in all_results:
        print(
            f"{result['variant']:<12}"
            f"{result['test_loss']:>10.4f}"
            f"{result['bleu']:>10.2f}"
            f"{result['chrf']:>10.2f}"
            f"{result['parameters']:>16,}"
        )

    if all_results:
        best_bleu = max(all_results, key=lambda x: x["bleu"])
        best_chrf = max(all_results, key=lambda x: x["chrf"])

        print(f"\nMeilleur BLEU : {best_bleu['variant'].upper()}")
        print(f"Meilleur chrF : {best_chrf['variant'].upper()}")

    print(f"\nRésultats JSON : {json_path}")
    print(f"Résultats CSV  : {csv_path}")
    print("Les exemples sont enregistrés dans le dossier results/.")


if __name__ == "__main__":
    main()
