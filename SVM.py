import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

#Chargement
train_df = pd.read_csv("train_split.csv")
test_df = pd.read_csv("test_split.csv")

#Verification des doublons
print("Verif des doublons : ")

# Recherche de textes strictement identiques entre train et test
# On transforme les colonnes en "sets" pour trouver l'intersection
set_train = set(train_df["Discours"])
set_test = set(test_df["Discours"])

doublons = set_train.intersection(set_test)
nb_doublons = len(doublons)

if nb_doublons > 0:
    print(f"{nb_doublons} textes identiques trouvés entre le train et le test !")
    test_df = test_df[~test_df["Discours"].isin(doublons)]
else:
    print("Aucun texte commun entre le train et le test.")

print(f"Taille finale - Train: {len(train_df)} | Test: {len(test_df)}\n")


X_train, y_train = train_df["Discours"], train_df["Parti"]
X_test, y_test = test_df["Discours"], test_df["Parti"]

#LinearSVC 
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LinearSVC(random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)

print("--- Résultats du modèle ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print("Classification Report :")
print(classification_report(y_test, y_pred))
