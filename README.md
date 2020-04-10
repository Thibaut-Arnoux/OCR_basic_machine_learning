# OCR_basic_machine_learning

[Tutoriel](https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning) : https://openclassrooms.com/fr/courses/4011851-initiez-vous-au-machine-learning

## Partie 1 : Le domaine de la Data Science

### Le cycle de travail

![](./images/cycle_travail.jpg)

- Récupération des données : source -> BDD, images, sons, textes, ...
- Nettoyer les données : S'assurer de la consistance des données, l'exclusion des valeurs aberrantes, gérer les valeurs manquantes
- Explorer les données :  Cette étape permet de mieux comprendre les différents comportements et de bien saisir le phénomène sous-jacent.  
Lorsque l'on a simplement besoin de comprendre ses données et les explorer, on peut faire appel à un **data analyst**. Ou bien un data analyst peut effectuer des études préliminaires avant de laisser le travail de modélisation au **data scientist**.
- Modélisez les données : C'est la création du modèle statistique associé aux données qui nous intéressent. C'est ce qu'on appelle le machine learning (ou apprentissage automatique).
- Evaluer & Interpréter : C'est connaitre la capacité qu'a notre modèle à représenter avec exactitude un phénomène, ou a minima sa capacité à résoudre la problématique. Cette étape doit s'accompagner de graphiques explicites.
- Mise en production : Rendre accessible l'exploitation de notre modèle via une API par exemple. Si la mise à l'échelle, le temps des calculs, le temps de réponses est trop long et qu'il faut une architecture plus poussée, on peut faire appel à un **data architect**

### L'étape de Modélisation

![](./images/etape_modelisation.jpg)


### Algorithme d'Apprentissage
- Régression Linéaire
- k-NN
- Support Vector Machine (SVM)
- Réseaux de neurones
- Random forests
- ...

### Types d'Apprentissage

- **Supervised learning** : On va récupérer des données dites annotées de leurs sorties pour entraîner le modèle, c'est-à-dire que les données sont déjà associées un label ou une classe cible et l'on veux que l'algorithme devienne capable, une fois entraîné, de prédire cette cible sur de nouvelles données non annotées.
- **Unsupervised learning** : Les données d'entrées ne sont pas annotées. L'algorithme d'entraînement s'applique dans ce cas à trouver seul les similarités et distinctions au sein de ces données, et à regrouper ensemble celles qui partagent des caractéristiques communes.
- **Semi-supervised** : L'algorithme prend en entrée certaines données annotées et d'autres non. Ce sont des méthodes très intéressantes qui tirent parti des deux mondes (supervised et unsupervised), mais bien sûr apportent leur lot de difficultés.
- **Reinforcement learning** : L'algorithme se base sur un cycle d'expérience / récompense et améliore les performances à chaque itération. Une analogie souvent citée est celle du cycle de dopamine : une "bonne" expérience augmente la dopamine et donc augmente la probabilité que l'agent répète l'expérience.

### Types de problématique

- **Classification Binaire** : On attend comme sortie une valeur binaire qui correspond à chacune de nos classes.
- **Classification Multi-label** : On attend comme sortie une probabilité d'appartenance à chaque classe.
- **Régression** : On attend comme sortie une valeur continue, un nombre.

## Partie 2 : Entrainement d'un modèle k-NN

## Partie 3 : Limites et problème du ML
