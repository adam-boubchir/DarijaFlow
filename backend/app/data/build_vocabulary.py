
from pathlib import Path

import pandas as pd

from vocabulary import Vocabulary, SOS_TOKEN, EOS_TOKEN


ROOT_DIR = Path(__file__).resolve().parents[3]

TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Charger uniquement les données d'entraînement
train_df = pd.read_csv(TRAIN_PATH)

english_vocab = Vocabulary()
darija_vocab = Vocabulary()

english_vocab.build(train_df["eng"].tolist())
darija_vocab.build(train_df["darija"].tolist())

# Sauvegarder les vocabulaires
english_vocab.save(ARTIFACTS_DIR / "english_vocab.json")
darija_vocab.save(ARTIFACTS_DIR / "darija_vocab.json")

print("Nombre de paires train :", len(train_df))
print("Taille du vocabulaire anglais :", len(english_vocab))
print("Taille du vocabulaire Darija :", len(darija_vocab))

print("\nTokens spéciaux :")
print("PAD =", english_vocab.stoi["<pad>"])
print("UNK =", english_vocab.stoi["<unk>"])
print("SOS =", english_vocab.stoi["<sos>"])
print("EOS =", english_vocab.stoi["<eos>"])

# Vérifier un exemple
example = train_df.iloc[0]

source_ids = english_vocab.encode(example["eng"])
target_ids = (
    [darija_vocab.stoi[SOS_TOKEN]]
    + darija_vocab.encode(example["darija"])
    + [darija_vocab.stoi[EOS_TOKEN]]
)

print("\nExemple anglais :", example["eng"])
print("Indices anglais :", source_ids)

print("\nExemple Darija :", example["darija"])
print("Indices Darija avec SOS/EOS :", target_ids)

print("\nRetour en tokens :")
print("Anglais :", english_vocab.decode(source_ids))
print("Darija :", darija_vocab.decode(target_ids))
