
from pathlib import Path

import pandas as pd
import sentencepiece as spm


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data" / "processed"
TOKENIZER_DIR = ROOT_DIR / "artifacts" / "sentencepiece"

VOCAB_SIZE = 2000

TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


def train_tokenizer(language: str, column: str):
    train_path = DATA_DIR / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Fichier d'entraînement introuvable : {train_path}"
        )

    df = pd.read_csv(train_path)

    if column not in df.columns:
        raise ValueError(
            f"Colonne '{column}' absente de {train_path}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # Corpus construit uniquement à partir du jeu d'entraînement.
    corpus_path = TOKENIZER_DIR / f"{language}_corpus.txt"

    with corpus_path.open("w", encoding="utf-8") as file:
        for sentence in df[column].fillna("").astype(str):
            sentence = sentence.strip()
            if sentence:
                file.write(sentence + "\n")

    prefix = TOKENIZER_DIR / language

    print(f"\nEntraînement du tokenizer : {language}")
    print(f"Corpus : {corpus_path}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<sos>",
        eos_piece="<eos>",
        hard_vocab_limit=False,
    )

    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")

    processor = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    print(f"Modèle sauvegardé : {model_path}")
    print(f"Vocabulaire réel : {processor.get_piece_size()} unités")
    print(f"Exemple : {processor.encode('hello how are you', out_type=str)}")

    return model_path, vocab_path


def main():
    train_tokenizer("english", "eng")
    train_tokenizer("darija", "darija")

    print("\nLes deux tokenizers ont été créés.")
    print(f"Dossier : {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
