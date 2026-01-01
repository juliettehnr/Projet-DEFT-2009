import pandas as pd
from parse_train import parse_train_parti
from parse_test import parse_test_parti
from sklearn.model_selection import train_test_split

# --- Appel des fonctions ---
train = parse_train_parti("./Projet DEFT/Corpus d_apprentissage/deft09_parlement_appr_fr.xml")

test = parse_test_parti(
    path_test= "./Projet DEFT/Corpus de test/deft09_parlement_test_fr.xml",
    path_ref = "./Projet DEFT/Données de référence/deft09_parlement_ref_fr.txt"
)

# --- Fusion train + test ---
# Le but étant de les mélanger pour faire un split 80/20 standard, plutôt que le split 60/40 pré-établi par le DEFT-2009

all_parties = {}

for party in train :
    all_parties[party] = (
        train.get(party, []) +
        test.get(party, [])
    )

texts = []
labels = []

for party, speeches in all_parties.items():
    texts.extend(speeches)
    labels.extend([party] * len(speeches))

df = pd.DataFrame({
    "Discours": texts,
    "Parti": labels
})

df = df.drop_duplicates(subset=["Discours"]) # On supprime les doublons

df = (
    df.groupby("Parti", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 2700), random_state=42)) # On limite à 2700 discours par classe (parce que la classe minoritaire avait 3347 discours contre plus de 11000 pour la classe majoritaire)
      .sample(frac=1, random_state=42)  # On mélange le dataframe et on fait en sorte qu'il soit reproductible
      .reset_index(drop=True)
)

# Vérification
"""
nb_doublons = df.duplicated(subset=["Discours"]).sum()
print(f"Nombre de doublons (Discours) : {nb_doublons}")
print("Taille totale :", len(df))
print(df["Parti"].value_counts())
"""

# --- Split du corpus ---

df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Parti"],
    random_state=42
)

# Vérification
"""
print("Taille totale :", len(df))
print("Train :", len(df_train))
print("Test :", len(df_test))

print(df_train["Parti"].value_counts())
print(df_test["Parti"].value_counts())
"""

# --- Création des fichiers CSV ---
df_train.to_csv(
    "train.csv",
    index=False,
    encoding="utf-8"
)

df_test.to_csv(
    "test.csv",
    index=False,
    encoding="utf-8"
)


