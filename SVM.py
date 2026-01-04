"""
Classification de Discours Politiques - Projet DEFT

Ce script implémente une chaîne de traitement complète pour prédire l'appartenance politique
d'un discours à l'aide d'un modèle LinearSVC.

Méthodologie :
1. Prétraitement textuel (minuscules, retrait des chiffres et caractères spéciaux).
2. Vectorisation TF-IDF avec n-grammes.
3. Entraînement d'un classifieur linéaire.
4. Évaluation via l'accuracy, le rapport de classification et une matrice de confusion.
"""
import pandas as pd
import re
import unicodedata
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
french_stopwords = stopwords.words("french")

# Chargement
train_df = pd.read_csv("./Projet DEFT/train.csv")
test_df = pd.read_csv("./Projet DEFT/test.csv")

# Nettoyage
def preprocess(text):
    text = str(text).lower()# passage en minuscules
    text = re.sub(r"\d+", "", text)# suppression de tous les chiffres
    text = re.sub(r"\W+", " ", text) #remplacement des caracteres spéciaux et ponctuation par un espace
    text = " ".join([w for w in text.split() if len(w) > 2])# on ne garde que les mots qui ont plus de 2 lettres pour éliminer le bruit
    return text.strip()

# Nettoyage des colonnes
train_df['clean_text'] = train_df['Discours'].apply(preprocess)
test_df['clean_text'] = test_df['Discours'].apply(preprocess)

# Vectorisation
vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),# analyse les mots seuls, mais aussi les suites de 2, 3 et 4 mots (pour capturer des slogans ou des noms de lois spécifiques à certains partis)
    max_features=100000, # permet d'avoir un vocabulaire très riche 
    min_df=3,# ignore les termes qui apparaissent dans moins de 3 discours (permet d'éliminer les fautes de frappe ou les mots trop rares qui n'aideraient pas à généraliser)
    max_df=0.85,#ignore les mots qui apparaissent dans plus de 85% des documents
    sublinear_tf=True,# réduit l'importance des mots répétés 50 fois par pur style oratoire (typique des discours politique)
    stop_words=french_stopwords # supprime les mots de liaison listés par NLTK
)

X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

y_train = train_df['Parti']
y_test = test_df['Parti']

# Modele
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

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title("Matrice de Confusion - SVM")
plt.tight_layout()
plt.savefig("confusion_matrix_deft_svm.png")