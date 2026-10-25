
from datasets import load_dataset

DATASET_NAME = "imomayiz/darija-english"
CONFIG_NAME = "sentences"

print("Chargement du dataset...")

dataset = load_dataset(
    DATASET_NAME,
    name=CONFIG_NAME,
)

print("\nConfigurations chargées :")
print(dataset)

for split_name, split_data in dataset.items():
    print(f"\n{'=' * 50}")
    print(f"Split : {split_name}")
    print(f"Nombre de lignes : {len(split_data)}")
    print(f"Colonnes : {split_data.column_names}")

    print("\nTrois exemples :")
    for example in split_data.select(range(min(3, len(split_data)))):
        print(example)

    print("\nInformations sur les colonnes :")
    for column in split_data.column_names:
        print(f"- {column}")
