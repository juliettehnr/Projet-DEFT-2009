import xml.etree.ElementTree as ET
import re
import csv
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuration des chemins
BASE = "/home/daguek/M2/Semestre1/App-art/proj/Projet DEFT"

PATH_XML_TRAIN = f"{BASE}/Corpus d_apprentissage/deft09_parlement_appr_fr.xml"
PATH_XML_TEST  = f"{BASE}/Corpus de test/deft09_parlement_test_fr.xml"
PATH_REF       = f"{BASE}/Données de référence/deft09_parlement_ref_fr.txt"

TRAIN_CSV = "train_fr_clean.csv"
TEST_CSV  = "test_fr_clean.csv"

# Clean le texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)       # supprime les balises XML
    text = re.sub(r"\d+", "", text)         # supprime les chiffres
    text = re.sub(r"[^\w\s]", "", text)     # supprime la ponctuation
    text = re.sub(r"\s+", " ", text)        # remplace espaces multiples par un espace
    return text.strip()

# Exports
def export_train_csv(xml_path, output_csv):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])

        for doc in root.findall(".//doc"):
            doc_id = doc.get("id")
            parti_elem = doc.find(".//PARTI")
            texte_elem = doc.find(".//texte")
            if parti_elem is None or texte_elem is None:
                continue

            text = clean_text(" ".join(texte_elem.itertext()))
            label = parti_elem.get("valeur")
            writer.writerow([doc_id, text, label])

    print(f"Train CSV généré : {output_csv}")

def export_test_csv(xml_path, ref_path, output_csv):
    label_map = {}
    with open(ref_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                label_map[parts[0]] = parts[1]

    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])

        for doc in root.findall(".//doc"):
            doc_id = doc.get("id")
            if doc_id not in label_map:
                continue

            texte_elem = doc.find(".//texte")
            if texte_elem is None:
                continue

            text = clean_text(" ".join(texte_elem.itertext()))
            label = label_map[doc_id]
            writer.writerow([doc_id, text, label])

    print(f"Test CSV généré : {output_csv}")

# génération des csv
export_train_csv(PATH_XML_TRAIN, TRAIN_CSV)
export_test_csv(PATH_XML_TEST, PATH_REF, TEST_CSV)

# Chargement des csv
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Stats")
print("Train :", len(train_df), Counter(train_df["label"]))
print("Test  :", len(test_df), Counter(test_df["label"]))

#Vérification des doublons 
train_texts_set = set(train_df["text"])
test_texts_set  = set(test_df["text"])
doublons= train_texts_set & test_texts_set
print("\nTextes identiques entre train et test :", len(doublons))

# Entrainement
vectorizer = TfidfVectorizer(ngram_range=(1,2), 
                             min_df=5, 
                             max_df=0.8)

X_train = vectorizer.fit_transform(train_df["text"])
X_test  = vectorizer.transform(test_df["text"])
y_train = train_df["label"]
y_test  = test_df["label"]
model = LinearSVC(class_weight="balanced", 
                  random_state=42, 
                  max_iter=2000)

model.fit(X_train, y_train)


# Evaluation
y_pred = model.predict(X_test)

print("Evaluation : ")
print(classification_report(y_test, y_pred))
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Pondérée : {f1_weighted:.4f}")

# matrice de conusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title("Matrice de Confusion - Identification des Partis (Français)")
plt.tight_layout()
plt.savefig("confusion_matrix_deft_fr.png")
print("Matrice de confusion sauvegardée : confusion_matrix_deft_fr.png")
