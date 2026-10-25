import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import sacrebleu
from tqdm import tqdm

from backend.app.data.translation_dataset import create_dataloader
from backend.app.models.seq2seq import Seq2Seq


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_experiment"
RESULTS_DIR = PROJECT_ROOT / "results_experiment"
METRICS_DIR = RESULTS_DIR / "metrics"
EXAMPLES_DIR = RESULTS_DIR / "examples"

BATCH_SIZE = 32
MAX_LENGTH = 50

MODEL_VARIANTS = ["vanilla", "wmc", "cifg"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID = 0
SOS_ID = 2
EOS_ID = 3


# ============================================================
# 2. PRÉPARATION DES DOSSIERS
# ============================================================

METRICS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DÉCODAGE DES SÉQUENCES
# ============================================================

def decode_generated(ids, tokenizer):
    """
    Convertit les identifiants générés en texte.

    Arrête le décodage à EOS et ignore PAD/SOS.
    """
    clean_ids = []

    for token_id in ids:
        token_id = int(token_id)

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, SOS_ID):
            continue

        clean_ids.append(token_id)

    return tokenizer.decode(clean_ids).strip()


def decode_reference(ids, tokenizer):
    """
    Convertit une séquence cible de référence en texte.
    Les séquences de référence contiennent normalement :
    SOS + tokens + EOS + éventuellement PAD.
    """
    clean_ids = []

    for token_id in ids:
        token_id = int(token_id)

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, SOS_ID):
            continue

        clean_ids.append(token_id)

    return tokenizer.decode(clean_ids).strip()


# ============================================================
# 4. CHARGEMENT D'UN MODÈLE
# ============================================================

def load_model(variant, english_tokenizer, darija_tokenizer):
    """
    Charge le meilleur checkpoint d'une variante.
    Aucun entraînement n'est effectué ici.
    """
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
        # Compatibilité avec les versions de PyTorch
        # qui ne possèdent pas l'argument weights_only.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    model_variant = checkpoint.get("variant", variant)

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
        cell_type=checkpoint.get(
            "cell_type",
            checkpoint.get("variant", variant),
        ),
        pad_idx=PAD_ID,
        sos_idx=SOS_ID,
        eos_idx=EOS_ID,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"\nModèle chargé : {variant.upper()}")
    print(f"Checkpoint : {checkpoint_path.name}")

    if "epoch" in checkpoint:
        print(f"Époque sélectionnée : {checkpoint['epoch']}")

    if "validation_chrf" in checkpoint:
        print(
            "chrF de validation : "
            f"{checkpoint['validation_chrf']:.4f}"
        )

    return model, checkpoint


# ============================================================
# 5. ÉVALUATION SUR LE JEU DE TEST
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    english_tokenizer,
    darija_tokenizer,
    variant,
):
    """
    Génère des traductions sur le jeu de test et calcule
    BLEU et chrF à l'échelle du corpus.
    """
    predictions = []
    references = []
    examples = []

    total_loss = 0.0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        reduction="sum",
    )

    start_time = time.time()

    for batch in tqdm(
        test_loader,
        desc=f"Test {variant.upper()}",
        leave=True,
    ):
        source = batch["source"].to(DEVICE)
        source_lengths = batch["source_lengths"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        # Calcul de la loss en mode teacher forcing désactivé.
        # Ce calcul sert uniquement d'indicateur complémentaire.
        logits = model(
            source=source,
            target=target,
            source_lengths=source_lengths,
            teacher_forcing_ratio=0.0,
        )

        target_for_loss = target[:, 1:]

        usable_length = min(
            logits.size(1),
            target_for_loss.size(1),
        )

        logits_for_loss = logits[:, :usable_length, :]
        target_for_loss = target_for_loss[:, :usable_length]

        loss_sum = criterion(
            logits_for_loss.reshape(
                -1, logits_for_loss.size(-1)
            ),
            target_for_loss.reshape(-1),
        )

        valid_tokens = (target_for_loss != PAD_ID).sum().item()

        total_loss += loss_sum.item()
        total_tokens += valid_tokens

        # Génération autorégressive des traductions.
        generated_batch = model.translate(
            source,
            source_lengths=source_lengths,
            max_length=MAX_LENGTH,
        )

        for index in range(source.size(0)):
            source_ids = source[index].detach().cpu().tolist()
            target_ids = target[index].detach().cpu().tolist()
            generated_ids = generated_batch[index].detach().cpu().tolist()

            source_text = english_tokenizer.decode(
                [
                    int(token_id)
                    for token_id in source_ids
                    if int(token_id) != PAD_ID
                ]
            ).strip()

            reference_text = decode_reference(
                target_ids,
                darija_tokenizer,
            )

            prediction_text = decode_generated(
                generated_ids,
                darija_tokenizer,
            )

            predictions.append(prediction_text)
            references.append(reference_text)

            # Conserver quelques exemples pour le rapport.
            if len(examples) < 100:
                examples.append(
                    {
                        "model": variant,
                        "english": source_text,
                        "reference_darija": reference_text,
                        "predicted_darija": prediction_text,
                    }
                )

    # Calcul corpus-level : toutes les phrases sont évaluées ensemble.
    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    ).score

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
    ).score

    average_loss = (
        total_loss / total_tokens
        if total_tokens > 0
        else None
    )

    elapsed_seconds = time.time() - start_time

    return {
        "model": variant,
        "test_bleu": float(bleu),
        "test_chrf": float(chrf),
        "test_loss": (
            float(average_loss)
            if average_loss is not None
            else None
        ),
        "test_sentences": len(references),
        "parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "evaluation_seconds": round(elapsed_seconds, 2),
    }, examples


# ============================================================
# 6. SAUVEGARDE DES RÉSULTATS
# ============================================================

def save_results(results, all_examples):
    """
    Sauvegarde les métriques au format CSV/JSON
    et les exemples au format CSV.
    """
    csv_path = METRICS_DIR / "test_comparison.csv"
    json_path = METRICS_DIR / "test_comparison.json"
    examples_path = EXAMPLES_DIR / "test_translation_examples.csv"

    # CSV des métriques
    fieldnames = [
        "model",
        "test_bleu",
        "test_chrf",
        "test_loss",
        "test_sentences",
        "parameters",
        "evaluation_seconds",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    # JSON des métriques
    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # CSV des exemples de traduction
    example_fields = [
        "model",
        "english",
        "reference_darija",
        "predicted_darija",
    ]

    with examples_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=example_fields,
        )
        writer.writeheader()
        writer.writerows(all_examples)

    print("\nFichiers sauvegardés :")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {examples_path}")


# ============================================================
# 7. PROGRAMME PRINCIPAL
# ============================================================

def main():
    print("=" * 65)
    print("ÉVALUATION FINALE DES MODÈLES ENGLISH -> DARIJA")
    print("=" * 65)

    print(f"Appareil utilisé : {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    # Charger le jeu de test, sans l'utiliser pour l'entraînement.
    test_loader, english_tokenizer, darija_tokenizer = (
        create_dataloader(
            split="test",
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    print(f"Nombre de batches de test : {len(test_loader)}")

    results = []
    all_examples = []

    for variant in MODEL_VARIANTS:
        model, checkpoint = load_model(
            variant,
            english_tokenizer,
            darija_tokenizer,
        )

        metrics, examples = evaluate_model(
            model=model,
            test_loader=test_loader,
            english_tokenizer=english_tokenizer,
            darija_tokenizer=darija_tokenizer,
            variant=variant,
        )

        results.append(metrics)
        all_examples.extend(examples)

        print(f"\nRésultats {variant.upper()} :")
        print(f"  BLEU test : {metrics['test_bleu']:.4f}")
        print(f"  chrF test : {metrics['test_chrf']:.4f}")

        if metrics["test_loss"] is not None:
            print(f"  Loss test : {metrics['test_loss']:.4f}")

        print(f"  Paramètres : {metrics['parameters']:,}")
        print(
            "  Temps d'évaluation : "
            f"{metrics['evaluation_seconds']:.2f} secondes"
        )

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    save_results(results, all_examples)

    # Classement descriptif selon chrF sur le jeu de test.
    # Ce classement ne doit pas servir à ajuster les modèles.
    ranking = sorted(
        results,
        key=lambda item: item["test_chrf"],
        reverse=True,
    )

    print("\n" + "=" * 65)
    print("COMPARAISON FINALE SUR LE JEU DE TEST")
    print("=" * 65)

    print(
        f"{'Modèle':<12}"
        f"{'BLEU':>12}"
        f"{'chrF':>12}"
        f"{'Paramètres':>18}"
    )

    print("-" * 54)

    for item in ranking:
        print(
            f"{item['model'].upper():<12}"
            f"{item['test_bleu']:>12.4f}"
            f"{item['test_chrf']:>12.4f}"
            f"{item['parameters']:>18,}"
        )

    print("\nÉvaluation terminée.")
    print(
        "Important : les métriques du test servent à rapporter "
        "les performances finales, pas à sélectionner de nouveau "
        "les checkpoints."
    )


if __name__ == "__main__":
    main()
