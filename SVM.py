import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

#Chargement
train_df = pd.read_csv("./Projet DEFT/train.csv")
test_df = pd.read_csv("./Projet DEFT/test.csv")

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
