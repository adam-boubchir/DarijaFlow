
from pathlib import Path

import pandas as pd

from backend.app.data.vocabulary import Vocabulary, tokenize


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def analyze_split(split, english_vocab, darija_vocab):
    path = DATA_DIR / f"{split}.csv"
    df = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE : {split.upper()}")
    print(f"{'=' * 60}")
    print(f"Nombre de paires : {len(df)}")

    for language, column, vocab in [
        ("Anglais", "eng", english_vocab),
        ("Darija", "darija", darija_vocab),
    ]:
        total_tokens = 0
        unknown_tokens = 0
        lengths = []

        for sentence in df[column].fillna("").astype(str):
            tokens = tokenize(sentence)
            lengths.append(len(tokens))
            total_tokens += len(tokens)
            unknown_tokens += sum(
                token not in vocab.stoi for token in tokens
            )

        percentage = (
            100 * unknown_tokens / total_tokens
            if total_tokens else 0
        )

        print(f"\n{language}")
        print(f"  Tokens totaux       : {total_tokens:,}")
        print(f"  Tokens inconnus     : {unknown_tokens:,}")
        print(f"  Taux de <unk>       : {percentage:.2f}%")
        print(
            "  Longueur moyenne    : "
            f"{sum(lengths) / max(len(lengths), 1):.2f} tokens"
        )
        print(f"  Phrase la plus longue : {max(lengths, default=0)} tokens")


def main():
    english_vocab = Vocabulary.load(
        ARTIFACTS_DIR / "english_vocab.json"
    )
    darija_vocab = Vocabulary.load(
        ARTIFACTS_DIR / "darija_vocab.json"
    )

    print("VOCABULAIRE ANGLAIS :", len(english_vocab))
    print("VOCABULAIRE DARIJA  :", len(darija_vocab))

    for split in ["train", "validation", "test"]:
        analyze_split(split, english_vocab, darija_vocab)


if __name__ == "__main__":
    main()
