import xml.etree.ElementTree as ET
import pandas as pd

# --- Création de listes vides pour chaque parti politique
ELDR = []
GUE_NGL = []
PPE_DE = []
PSE = []
VERTS_ALE = []

# --- Mapping parti -> liste ---
party_lists = {
    "ELDR": ELDR,
    "GUE-NGL": GUE_NGL,
    "PPE-DE": PPE_DE,
    "PSE": PSE,
    "Verts-ALE": VERTS_ALE
}

# --- Lecture du fichier txt (id -> parti) ---
id_to_party = {}

with open("./Projet DEFT/Données de référence/deft09_parlement_ref_fr.txt", "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()

        # On ignore les lignes vides
        if not line:
            continue

        # On vérifie qu'il y a bien 2 colonnes
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"Ligne ignorée ({line_number}) : {line}")
            continue

        doc_id, party = parts
        id_to_party[doc_id] = party


# --- Parsing du XML ---
tree = ET.parse('./Projet DEFT/Corpus de test/deft09_parlement_test_fr.xml')
root = tree.getroot()


for doc in root.findall("doc"):
    doc_id = doc.get("id")

    if doc_id in id_to_party:
        party = id_to_party[doc_id]

       # On vérifier que le parti est dans notre mapping
        if party in party_lists:

            # --- Extraire tout le texte des balises <p> à l'intérieur de <doc> ---
            speech = " ".join(
                " ".join(p.itertext()).strip()
                for p in doc.findall(".//p")
                if p.text and p.text.strip()
            )

            if speech:  # ne pas ajouter de chaînes vides
                party_lists[party].append(speech)

# --- Création du DataFrame Pandas ---
data = [{'Parti': 'ELDR', 'Discours' : discours} for discours in ELDR] + [{'Parti': 'GUE-NGL', 'Discours' : discours} for discours in GUE_NGL] + [{'Parti': 'PPE-DE', 'Discours' : discours} for discours in PPE_DE] + [{'Parti': 'PSE', 'Discours' : discours} for discours in PSE] + [{'Parti': 'Verts-ALE', 'Discours' : discours} for discours in VERTS_ALE]

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
