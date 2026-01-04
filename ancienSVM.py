"""
Classification de Discours Politiques - Projet DEFT

NOTE IMPORTANTE : 
C'est grâce à ce script et à l'analyse des données que nous avons identifié la présence de doublon entre les 
données d'entraînement et les données de test. Cette découverte a été le point  de départ de notre réflexion 
sur la nécessité d'un nettoyage plus rigoureux pour éviter les biais d'évaluation.

Fonctionnalités :
1. Parsing des fichiers XML et alignement avec le fichier de référence.
2. Nettoyage de texte (Regex, suppression XML, normalisation).
3. Exportation des données en format CSV structuré.
4. Analyse statistique des classes et détection des doublons.
5. Vectorisation TF-IDF (unigrammes/bigrammes) et classification via LinearSVC.
6. Évaluation des performances (F1-score, Rapport de classification).
7. Génération et sauvegarde d'une matrice de confusion.
"""

import xml.etree.ElementTree as ET
import re
import csv
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuration des chemins
BASE = "/home/daguek/M2/Semestre1/App-art/proj/Projet DEFT"

PATH_XML_TRAIN = f"{BASE}/Corpus d_apprentissage/deft09_parlement_appr_fr.xml"
PATH_XML_TEST  = f"{BASE}/Corpus de test/deft09_parlement_test_fr.xml"
PATH_REF       = f"{BASE}/Données de référence/deft09_parlement_ref_fr.txt"

TRAIN_CSV = "train_fr_clean.csv"
TEST_CSV  = "test_fr_clean.csv"

# Clean le texte
def preprocess(text):
    text = str(text).lower()# passage en minuscules
    text = re.sub(r"\d+", "", text)# suppression de tous les chiffres
    text = re.sub(r"\W+", " ", text) #remplacement des caracteres spéciaux et ponctuation par un espace
    text = " ".join([w for w in text.split() if len(w) > 2])# on ne garde que les mots qui ont plus de 2 lettres pour éliminer le bruit
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

            text = preprocess(" ".join(texte_elem.itertext()))
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

            text = preprocess(" ".join(texte_elem.itertext()))
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
vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),# analyse les mots seuls, mais aussi les suites de 2, 3 et 4 mots (pour capturer des slogans ou des noms de lois spécifiques à certains partis)
    max_features=100000, # permet d'avoir un vocabulaire très riche 
    min_df=3,# ignore les termes qui apparaissent dans moins de 3 discours (permet d'éliminer les fautes de frappe ou les mots trop rares qui n'aideraient pas à généraliser)
    max_df=0.85,#ignore les mots qui apparaissent dans plus de 85% des documents
    sublinear_tf=True,# réduit l'importance des mots répétés 50 fois par pur style oratoire (typique des discours politique)
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test  = vectorizer.transform(test_df["text"])
y_train = train_df["label"]
y_test  = test_df["label"]

model = LinearSVC(
    C=0.5, # force le modèle à ne pas trop se coller aux  détails spécifiques du train pour mieux géneraliser sur les nouveaux discours    
    max_iter=6000,#définit le nombre maximum de tentatives du modèle pour trouver la solution optimale
    random_state=42 # garantit d'obtenir exactement les mêmes scores d'accuracy à chaque fois          
)

model.fit(X_train, y_train)

# Prédiction et Evauation
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# matrice de conusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title("Matrice de Confusion - Identification des Partis (Français)")
plt.tight_layout()
plt.savefig("confusion_matrix_deft_fr.png")
print("Matrice de confusion sauvegardée : confusion_matrix_deft_fr.png")
