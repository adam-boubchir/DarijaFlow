
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# 1. Chemins du projet
ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 2. Charger le dataset
print("Chargement du dataset...")
dataset = load_dataset(
    "imomayiz/darija-english",
    name="sentences",
)

df = dataset["sentences"].to_pandas()

print(f"\nDimensions initiales : {df.shape}")
print(f"Colonnes : {df.columns.tolist()}")

# 3. Sauvegarder une copie brute, sans modification
raw_path = RAW_DIR / "darija_english_raw.csv"
df.to_csv(raw_path, index=False, encoding="utf-8")
print(f"\nCopie brute sauvegardée : {raw_path}")

# 4. Valeurs manquantes
print("\n--- Valeurs manquantes ---")
print(df.isna().sum())

# 5. Chaînes vides ou composées uniquement d'espaces
print("\n--- Chaînes vides ---")

for column in ["eng", "darija", "darija_ar"]:
    empty_mask = (
        df[column]
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
    )
    print(f"{column}: {int(empty_mask.sum())}")

# 6. Doublons exacts sur les paires utilisées
print("\n--- Doublons ---")
print(
    "Paires eng-darija dupliquées :",
    int(df.duplicated(subset=["eng", "darija"]).sum()),
)
print(
    "Phrases anglaises répétées :",
    int(df["eng"].duplicated().sum()),
)

# 7. Statistiques de longueur
print("\n--- Longueur des phrases en mots ---")

for column in ["eng", "darija"]:
    lengths = (
        df[column]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print(f"\n{column}:")
    print(lengths.describe(percentiles=[0.50, 0.90, 0.95, 0.99]))

# 8. Afficher quelques exemples problématiques
print("\n--- Exemples avec anglais ou Darija manquant ---")
missing_mask = df[["eng", "darija"]].isna().any(axis=1)
print(df.loc[missing_mask, ["eng", "darija"]].head(10))

print("\nAnalyse terminée. Aucune ligne n'a été supprimée.")
