# DEFT’09 : Détection du parti politique de l’orateur dans un corpus de discours de députés au Parlement Européen

Contributrices : Juliette HENRY & Keren DAGUE

## Abstract

Nous avons élaboré une chaîne de traitement
pour classifier automatiquement des discours
politiques selon le parti de leurs orateurs. Cette
chaine pré-traite le corpus écrit et le soumet
à un modèle d’apprentissage fine-tuné. Notre
modèle de classification LinearSVC obtient une
accuracy de 54,14%, des résultats mitigés mais
satisfaisants par rapport à la difficulté de la
tâche.

## Introduction

Cet article fait état des résultats obtenus pour la
tâche n°3 du DÉfi Fouille de Texte de l’édition
2009 (1). Cette année-là, seule l’Université de
Montréal avait soumis leurs résultats pour cette
tâche. Elle consistait à prédire le parti politique
des députés européens selon leurs discours au Par-
lement. Dominic Forest & al. (2) avaient obtenu,
avec un classifieur Naive Bayes, des scores de rap-
pel de 33,30% et de précision de 33,50%. Notre
premier objectif était de vérifier si notre mod-
èle Naive Bayes surpassait les résultats obtenus
par l’Université de Montréal en 2009. Notre sec-
ond était de voir quel modèle pouvait obtenir de
meilleurs résultats.

## Présentation du corpus

### Le corpus DEFT’09

Le corpus parallèle fourni par le DEFT’09 rassem-
ble des discours politiques issus de débats de
députés au Parlement Européen. Ces discours da-
tent d’entre 1999 et 2004. Le défi prévoyait trois
langues : l’anglais, le français et l’italien. Étant
donné que les chercheurs de l’Université de Mon-
tréal ont choisi de ne traiter que le sous-corpus en
français, et que notre but est de comparer leurs ré-
sultats aux notres, nous avons décidé de faire de
même. Celui-ci prend la forme d’un dossier con-
tenant d’une part les données d’entrainement au
format XML (60% du corpus initial), et d’autre
part les données de test au format XML (40% du
corpus initial), ainsi qu’un fichier texte comprenant
les noms des partis politiques associés aux discours
du test.

Les partis retenus pour le défi sont au nom-
bre de cinq : ELDR (Groupe du Parti européen
des libéraux, démocrates et réformateurs), GUE-
NGL (Groupe confédéral de la Gauche unitaire eu-
ropéenne/Gauche verte nordique), PPE-DE (Parti
populaire européen & Démocrates Européens),
PSE (Groupe Socialiste au Parlement Européen)
et Verts-ALE (Groupe des Verts/Alliance libre eu-
ropéenne). La répartition des discours dans ces
classes sont données dans la table 1.

Table 1: Répartition des discours par parti politique dans
le corpus DEFT’09

### Prétraitement des données

Au début de notre projet, nous avons développé un
premier script qui affichait des résultats supérieurs
à nos attentes, avec une accuracy de 80%. Ce
score nous a semblé anormalement élevé pour cette
tâche. Nous avons donc fait le choix de fouiller
concrètement dans les données brutes, nous avons
découvert la présence de doublons entre les fichiers
d’entraînement et de test. Pour corriger ce prob-
lème, nous avons procédé différemment en mettant
en place un nettoyage des données avant de lancer
nos scripts finaux.

Par la suite nous avons donc décider de crée deux
scripts permettant de parser les fichiers XML de
train et de test. Le premier script parse_train.py
parse le fichier de train en récupérrant les discours
et les partis contenus dans les balises. Il renvoie un
dictionnaire python. Le second script parse_test.py
fait la même chose avec le fichier de test mais va
chercher les labels des partis dans le fichier txt.

Ces deux dictionnaires sont ensuite appelés
dans un troisième script dataframe.py qui mélange
les deux dictionnaires et supprime les doublons.
Comme le split 60/40 de base ne pouvait plus être
respecté avec la suppression des doublons, nous en
avons profité pour séparer les données en 80/20, la
proportion standard de nos jours. À la fin, nous
obtenons un corpus de 25432 textes dont 20345
sont réservés au train et 5087 au test. La répartition
des classes est visible dans la table 2.

Table 2: Répartition des discours par parti politique dans
notre corpus néttoyé 

La dernière partie du prétraitement se fait di-
rectement dans le script de nos modèles NB.py
et SVM.py. Avant la vectorisation, les scripts
s’occupent de nettoyer le corpus, notamment en
retirant les chiffres, en remplaçant les majus-
cule par des minuscules et les caracteres spéci-
aux/ponctuations par un espace et en supprimant
les mots de moins de deux lettres. La chaîne de
traitement est résumée dans la figure 1.

Figure 1: Chaîne de traitement des données

## Nos modèles de classification

Forest & al. (2009) avaient opté pour un modèle
Naive Bayes. Nous avons fait le choix d’entraîner
deux modèles : un MultinomialNB et un Lin-
earSVC. Le premier nous sert à comparer les per-
formances d’un même modèle à plus de 15 ans
d’interval. Le second est notre modèle de choix
pour cette tâche. En effet, nous avions déjà tra-
vaillé en avril 2025 sur un projet très similaire qui
consistait aussi à classifier des discours politiques
provenant des députés de l’Assemblée Nationale.
Nous avions à cette époque obtenus des résultats
corrects avec 63,16% d’accuracy, d’où notre choix
de réitérer l’expérience avec un modèle SVM.

Nous avons choisi une vectorisation Count pour
le modèle Naive Bayes et une vectorisation TF-
IDF pour le SVM. Nous avons fine-tuné les deux
modèles dans le but d’obtenir de meilleurs résultats.
Ainsi, pendant la vectorisation, les deux scripts
s’occupent de retirer les stopwords NLTK français
ainsi que les mots qui apparaissent dans trop (plus
de 85% du corpus) ou pas assez (moins de 3 textes)
de documents. De plus, la vectorisation des scripts
NB.py et SVM.py sélectionne respectivement des
n-grams allant de 1 à 3 et de 1 à 4. C’est-à-dire que
le modèle va analyser le mot seul mais également
les suites de 2, 3 et 4 mots (pour SVM), ce qui
est utile pour capturer des slogans ou des noms
de lois spécifiques à certains partis. Enfin, pour
le modèle SVM, nous avons activé la pondération
sublinear_tf. Ce choix est stratégique pour le
corpus politique : il permet de lisser l’importance
d’un mot répété de nombreuses fois par pur effet
de style oratoire. Ainsi, un terme mentionné 50
fois n’aura pas dix fois plus de poids qu’un terme
mentionné 5 fois, évitant ainsi que le style d’un seul
orateur ne biaise trop fortement la classification.

En ce qui concerne la modélisation, pour le
modèle LinearSVC, nous avons fixé le paramètre
de régularisation à 0.5. En utilisant une valeur
inférieure à 1.0, nous imposons une contrainte
de régularisation plus forte. Cela force le mod-
èle à ne pas sur-apprendre les spécificités ou le
"bruit" des données d’entraînement, évitant ainsi
l’overfitting. De plus, nous avons augmenté le nom-
bre d’itérations (max_iter=6000) pour garantir la
convergence du modèle vers une solution optimale.

Pour le modèle MultinomialNB, nous avons
ajusté le paramètre de lissage alpha à 0.1.
Ce réglage permet d’attribuer une probabilité
minimale aux termes absents du vocabulaire
d’apprentissage sans pour autant masquer les spé-
cificités lexicales propres à chaque parti.

Enfin, pour assurer la reproductibilité de
nos expériences, nous avons fixé le paramètre
random_state à 42. Ceci garantit que les mesures
de performance obtenues sont strictement iden-
tiques à chaque exécution du script, facilitant ainsi
l’analyse comparative de nos différents tests.

## Nos Résultats

L’analyse des résultats obtenus avec le modèle
SVM révèle une accuracy globale de 54,14%. Le
rapport de classification souligne une disparité im-
portante entre les classes, le groupe GUE-NGL 
obtenant la meilleure performance avec un F1-score
de 0,64 et une précision de 0,67. En revanche, nous
observons une faiblesse majeure dans le rappel des
classes minoritaires comme ELDR (0,23) et Verts-
ALE (0,29). La matrice de confusion du SVM
confirme cette difficulté : le modèle prédit correcte-
ment 1 290 documents pour le PPE-DE, mais il
confond massivement le PSE avec ce dernier, avec
545 erreurs de prédiction. Cette tendance montre
que le SVM, bien que performant sur les classes
bien définies, se laisse influencer par le volume des
données des partis majoritaires.

Table 3: Résultats de rappel, précision et accuracy des
deux modèles

Pour remédier à cela nous avions testé
l’utilisation de l’hyperparamètre class_weight=
’balanced’. Cette option permet de compenser le
déséquilibre du support en augmentant le poids des
classes minoritaires comme l’ELDR ou les Verts-
ALE. Cet ajustement faisait chuter notre accuracy
globale. En effet, en tentant de mieux capter les
petits partis, le modèle commettait beaucoup plus
d’erreurs sur les classes majoritaires (PPE-DE et
PSE), qui représentent la plus grande partie du cor-
pus.

Le modèle MultinomialNB, avec une accuracy
de 49,56%, souffre plus nettement des déséquili-
bres du corpus. Sa matrice de confusion met en év-
idence un phénomène de saturation vers les classes
dominantes : le groupe PPE-DE attire 1 137 bonnes
prédictions, mais devient un "puits" pour les erreurs
des autres partis. On remarque notamment que 500
documents appartenant au PSE sont classés à tort
en PPE-DE. Cette confusion massive suggère que
les probabilités a priori du modèle Naive Bayes
favorisent systématiquement les classes au support
le plus élevé lorsque le lexique employé est trop
institutionnel pour être discriminant.

Pour tenter de remédier à ce déséquilibre, nous
avons implémenté le modèle ComplementNB, une
variante spécifiquement conçue pour corriger les
biais liés aux corpus asymétriques. Cependant,
bien que ce modèle soit théoriquement plus robuste
face aux classes dominantes, nos tests ont révélé
une baisse de l’accuracy globale.

## Conclusion

Cette étude nous a permis de réévaluer les per-
formances de modèles de classification classiques
sur le corpus DEFT’09. Nos modèles ont montré
une nette progression par rapport aux travaux de
référence de 2009. Le modèle LinearSVC s’est
avéré être le plus performant avec une accuracy
de 54%, dépassant significativement les scores de
rappel et de précision de 33% obtenus initialement
par l’Université de Montréal. Notre modèle Multi-
nomialNB obtient de meilleures performances que
celui utilisé par l’Université de Montréal pour cette
même tâche. Ces résultats confirment l’efficacité
des approches par vecteurs de support pour la clas-
sification de textes politiques à haute dimension-
nalité.

Dans une optique d’amélioration des résultats,
qui sont encore trop proches du hasard, il serait
intéressant d’opérer d’autres traitements sur le
corpus, comme la lemmatisasion par exemple.
Une autre possibilité serait d’utiliser des modèles
d’apprentissage profond tel que BERT pour com-
parer leurs performances.

## References

[1] Grouin & al. Présentation de l’édition 2009 du
DÉfi Fouille de Textes (DEFT’09). DEFT’09 "DÉfi
Fouille de Textes", Atelier de clôture, Jun 2009, Paris,
France.

[2] Forest & al. Impacts de la variation du nombre de
traits discriminants sur la catégorisation des docu-
ments DEFT’09 "DÉfi Fouille de Textes", Atelier de
clôture, Jun 2009, Paris, France.
