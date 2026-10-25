
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------
# 1. Chemins
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]

RAW_PATH = ROOT_DIR / "data" / "raw" / "darija_english_raw.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Charger les données brutes
# --------------------------------------------------
df = pd.read_csv(RAW_PATH)

print("Dimensions initiales :", df.shape)

# On conserve seulement les colonnes utiles à la traduction
df = df[["eng", "darija"]].copy()

# --------------------------------------------------
# 3. Nettoyage de base
# --------------------------------------------------
for column in ["eng", "darija"]:
    df[column] = df[column].astype("string").str.strip()

df = df.dropna(subset=["eng", "darija"])

df = df[
    df["eng"].ne("")
    & df["darija"].ne("")
].copy()

before_duplicates = len(df)

# Supprimer uniquement les paires exactes répétées
df = df.drop_duplicates(subset=["eng", "darija"]).reset_index(drop=True)

print("Paires valides avant déduplication :", before_duplicates)
print("Paires conservées :", len(df))
print("Paires exactes supprimées :", before_duplicates - len(df))

# --------------------------------------------------
# 4. Vérifier les phrases anglaises distinctes
# --------------------------------------------------
print("Phrases anglaises distinctes :", df["eng"].nunique())

# --------------------------------------------------
# 5. Split par phrase anglaise
# --------------------------------------------------
# Les traductions d'une même phrase anglaise restent
# ensemble dans un seul split.
splitter_test = GroupShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=42,
)

train_val_idx, test_idx = next(
    splitter_test.split(df, groups=df["eng"])
)

train_val = df.iloc[train_val_idx].reset_index(drop=True)
test = df.iloc[test_idx].reset_index(drop=True)

splitter_val = GroupShuffleSplit(
    n_splits=1,
    test_size=1 / 9,
    random_state=42,
)

train_idx, val_idx = next(
    splitter_val.split(train_val, groups=train_val["eng"])
)

train = train_val.iloc[train_idx].reset_index(drop=True)
val = train_val.iloc[val_idx].reset_index(drop=True)

# --------------------------------------------------
# 6. Vérifier l'absence de fuite entre les splits
# --------------------------------------------------
train_groups = set(train["eng"])
val_groups = set(val["eng"])
test_groups = set(test["eng"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# --------------------------------------------------
# 7. Sauvegarder les données préparées
# --------------------------------------------------
train.to_csv(PROCESSED_DIR / "train.csv", index=False, encoding="utf-8")
val.to_csv(PROCESSED_DIR / "validation.csv", index=False, encoding="utf-8")
test.to_csv(PROCESSED_DIR / "test.csv", index=False, encoding="utf-8")

print("\n--- Résultats des splits ---")
print(f"Train      : {len(train)} lignes ({len(train) / len(df):.1%})")
print(f"Validation : {len(val)} lignes ({len(val) / len(df):.1%})")
print(f"Test       : {len(test)} lignes ({len(test) / len(df):.1%})")

print("\nFichiers sauvegardés dans :", PROCESSED_DIR)
print("Vérification des groupes : aucune phrase anglaise partagée entre les splits.")
