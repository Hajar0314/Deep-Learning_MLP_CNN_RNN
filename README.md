# Projet Deep Learning académique

## Présentation générale
Ce projet a été réalisé dans le cadre d'un module académique dédié au Deep Learning. L'objectif est d'étudier, implémenter et comparer plusieurs architectures de réseaux de neurones en utilisant l'écosystème PyTorch. Le travail est structuré en trois parties principales : un modèle perceptron multi-couches (MLP), un réseau de neurones convolutionnel (CNN) et un modèle récurrent / seq2seq (RNN).

## Objectif global
- Étudier les principes fondamentaux des architectures MLP, CNN et RNN.
- Implémenter ces modèles en Python avec PyTorch.
- Expérimenter différents hyperparamètres.
- Évaluer et comparer les performances sur des tâches représentatives.
- Développer une capacité d'analyse critique des résultats obtenus.

## Objectifs pédagogiques
- Comprendre les fondements théoriques des architectures étudiées.
- Implémenter des modèles de Deep Learning en Python et PyTorch.
- Maîtriser la préparation et l'analyse des données.
- Expérimenter l'impact des hyperparamètres sur l'entraînement.
- Évaluer les performances et comparer différentes approches.
- Tirer des enseignements des résultats et proposer des améliorations.

## Structure des dossiers et des fichiers
```
DL-Pojet/
├── .gitignore
├── .python-version
├── .venv/
├── README.md
├── main.py
├── pyproject.toml
├── CNN/
│   ├── Notebook_Partie 2_ CNN.ipynb
│   ├── ablation_results.png
│   ├── confusion_matrices.png
│   ├── corr2d_filters.png
│   ├── data/
│   │   └── FashionMNIST/
│   │       └── raw/
│   │           ├── t10k-images-idx3-ubyte
│   │           ├── t10k-labels-idx1-ubyte
│   │           ├── train-images-idx3-ubyte
│   │           └── train-labels-idx1-ubyte
│   ├── fashion_mnist_samples.png
│   ├── feature_map_0.png
│   ├── feature_map_1.png
│   ├── feature_map_2.png
│   ├── feature_map_3.png
│   ├── learned_filters_conv1.png
│   ├── LeNetModern_—_Fashion-MNIST.png
│   ├── mlp_vs_cnn.png
│   ├── params_comparison.png
│   ├── pooling_comparison.png
│   └── synthese_finale.png
├── MLP/
│   ├── Notebook _Partie 1_ MLP.ipynb
│   ├── best_mlp_custom.pth
│   ├── best_mlp_sequential.pth
│   └── partie1_visualisations.png
├── RNN/
│   ├── Notebook_Partie 3_ RNN.ipynb
│   ├── best_seq2seq.pt
│   ├── bptt_clipping.png
│   ├── comparaison_complete.png
│   ├── comparaison_params.png
│   ├── distribution_longueurs.png
│   ├── embedding_visualisation.png
│   ├── evaluation_bleu.png
│   ├── fra_eng_data/
│   │   └── fra.txt
│   ├── perplexite_analyse.png
│   ├── regle_chaine.png
│   ├── seq2seq_training.png
│   └── synthese_finale.png
├── Rapport/Deep_Learning_Rapport_MLP_CNN_RNN_Hajar_Belhachmi_G2
└── uv.lock
```

### Rôles des dossiers et fichiers
- `README.md` : documentation générale du projet.
- `pyproject.toml` : configuration du projet et liste des dépendances.
- `main.py` : script principal ou point d'entrée potentiel pour tests et exécutions globales.
- `MLP/` : notebook, modèles sauvegardés et visualisations pour la partie perceptron multi-couches.
- `CNN/` : notebook, données Fashion-MNIST, figures et résultats pour la partie convolutionnelle.
- `RNN/` : notebook, modèle seq2seq sauvegardé, corpus de traduction et visualisations pour la partie récurrente.
- `Rapport/` : espace dédié au rapport académique et aux synthèses écrites.
- `.venv/` : environnement virtuel Python localisé.

## Technologies et outils utilisés
- Python
- PyTorch
- Torchvision
- Torchtext
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- SacreBLEU
- tqdm

## Description détaillée des notebooks

### Notebook 1 : MLP (Multi-Layer Perceptron)
Objectifs de la partie MLP : étudier les architectures fully-connected, comprendre la propagation avant et arrière, et mesurer le comportement du perceptron multi-couches sur une tâche de classification.

Principales sections :
- Préparation des données : chargement, normalisation et transformation des entrées.
- Définition de l'architecture MLP : couches linéaires, fonctions d'activation et structure du modèle.
- Entraînement : boucle d'entraînement, calcul de la perte et mise à jour des poids.
- Évaluation : calcul de la précision, matrice de confusion et suivi des performances sur ensemble de validation.
- Visualisation : courbes d'apprentissage, comparaison architectures et interprétation des résultats.

Rôle de chaque section :
- Préparation des données : garantir la qualité des entrées pour l'entraînement.
- Architecture MLP : formaliser le modèle et ses hyperparamètres.
- Entraînement : faire apprendre le réseau aux données.
- Évaluation : quantifier la performance et détecter d'éventuels surapprentissages.
- Visualisation : communiquer graphiquement les résultats.

### Notebook 2 : CNN (Convolutional Neural Networks)
Objectifs de la partie CNN : démontrer la supériorité des réseaux convolutionnels sur les tâches de classification d'images, analyser les opérations de convolution et pooling, et comparer les performances avec un MLP.

Sections principales :
- Chargement de Fashion-MNIST : présentation du dataset et des exemples d'images.
- Prétraitement : normalisation, transformées et préparation des batches.
- Convolutions et pooling : explication des filtres, des cartes de caractéristiques et des opérations de sous-échantillonnage.
- Architecture LeNet / CNN moderne : définition du modèle convolutionnel et de sa structure.
- Entraînement et évaluation : optimisation, calcul de la perte, précision et analyse des résultats.
- Visualisation de cartes de caractéristiques : exploration des filtres appris et des activations internes.
- Comparaison MLP/CNN : synthèse des performances et des comportements.

Explication de chaque partie :
- Fashion-MNIST : dataset d'images de vêtements pour classification.
- Convolutions : construction des représentations locales et invariantes.
- Pooling : réduction de dimension et robustesse.
- LeNet : architecture CNN classique adaptée à Fashion-MNIST.
- Visualisations : compréhension du rôle des filtres et des activations.
- Comparaison : mise en évidence des avantages du CNN sur des données visuelles.

### Notebook 3 : RNN (Recurrent Neural Networks)
Objectifs de la partie RNN : étudier la modélisation séquentielle du langage, implémenter des modèles récurrents et séquence-à-séquence, puis évaluer les résultats sur une tâche de traduction.

Sections principales :
- Présentation du corpus : préparation des données de traduction français-anglais.
- Analyse des séquences : distribution des longueurs et gestion des vocabulaires.
- Embeddings : représentation dense des mots.
- Modèles RNN, LSTM, GRU : comparaison des architectures récurrentes.
- Seq2Seq : construction d'un modèle encodeur-décodeur pour traduction.
- Stratégies de décodage : décodage en mode greedy et autres méthodes.
- Évaluation : perplexité, BLEU score et comparaison des performances.

Rôle de chaque partie :
- Corpus de traduction : définir le problème de modélisation du langage.
- Analyse des séquences : comprendre les contraintes des entrées et sorties.
- Embeddings : transformer les mots en vecteurs exploitables.
- RNN/LSTM/GRU : choisir une architecture adaptée aux dépendances temporelles.
- Seq2Seq : mettre en œuvre un modèle capable de traduire phrase par phrase.
- Décodage : générer du texte à partir du modèle entraîné.
- Évaluation : mesurer la qualité linguistique et la cohérence des prédictions.

## Jeux de données utilisés
- Fashion-MNIST
  - Description : images 28x28 de vêtements en 10 classes.
  - Problème : classification d'images.
- Corpus français-anglais (`fra.txt`)
  - Description : paires de phrases françaises et anglaises pour traduction.
  - Problème : modélisation du langage / traduction automatique.

## Résultats et analyses
- Le MLP permet d'aborder une première mise en œuvre de réseau de neurones, mais ses capacités restent limitées sur des données structurées complexes.
- Le CNN montre une meilleure performance sur Fashion-MNIST grâce à la capacité à extraire des caractéristiques locales et invariantes.
- Le RNN / Seq2Seq est adapté à la traduction, avec des métriques comme la perplexité et le BLEU pour évaluer la qualité des séquences générées.
- Les principales différences observées :
  - MLP : adapté aux entrées vectorisées et aux problèmes simples, mais moins efficace pour les images.
  - CNN : performant pour la classification d'images grâce aux convolutions et au pooling.
  - RNN : adapté aux données séquentielles et à la génération de texte, mais plus coûteux en temps d'entraînement.
- Enseignements tirés :
  - Le choix de l'architecture dépend fortement de la nature des données.
  - L'expérimentation d'hyperparamètres est essentielle pour optimiser les performances.
  - La visualisation des résultats aide à interpréter les modèles et à justifier les conclusions.

## Instructions d'exécution
### Prérequis
- Python 3.14 ou supérieur.
- Environnement virtuel activé (`.venv` recommandé).
- Installer les dépendances listées dans `pyproject.toml`.

### Installation des dépendances
Si vous utilisez `pip` :
```bash
pip install -r requirements.txt
```

ou, si vous utilisez Poetry / uv :
```bash
uv install
```

### Ordre recommandé d'exécution
1. `MLP/Notebook _Partie 1_ MLP.ipynb`
2. `CNN/Notebook_Partie 2_ CNN.ipynb`
3. `RNN/Notebook_Partie 3_ RNN.ipynb`

### Conseils pour reproduire les expériences
- Activez l'environnement virtuel avant de lancer les notebooks.
- Vérifiez que le dossier `CNN/data/FashionMNIST/raw` contient bien les fichiers du dataset.
- Utilisez des versions cohérentes des bibliothèques pour éviter les incompatibilités.
- Lancez chaque notebook de bout en bout et suivez les sections dans l'ordre proposé.
- Conservez les modèles sauvegardés (`.pth`, `.pt`) pour comparer rapidement les performances.
- Ajustez les hyperparamètres (learning rate, batch size, nombre d'époques) pour reproduire et prolonger les expérimentations.

## Conclusion
Ce projet offre une vue d'ensemble pédagogique du Deep Learning appliqué à trois grandes familles de modèles : MLP, CNN et RNN. Il combine implémentation, expérimentation, visualisation et évaluation pour renforcer la compréhension des architectures et de leur application à des tâches de classification d'images et de traduction automatique.