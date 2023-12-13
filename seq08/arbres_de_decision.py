# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Les arbres de décision et forêts aléatoires

# %% [markdown] editable=true slideshow={"slide_type": "skip"}
# ## TODO
#
# - [x] Jupytext
# - [ ] écrire un draft du notebook, y mettre les références bibliographiques, etc.
# - [ ] écrire des datasets (gpt)
# - [ ] implémenter un exemple d'apprentissage et d'inférence avec sklearn, tester sur les datasets précédents
# - [ ] faire des illustarations bouillon (récup img google, scan, dessins faits avec la tablette graphique, dalle, etc.)
# - [ ] lire les deux autres livres
# - [ ] implémenter un exemple d'apprentissage et d'inférence d'ID3 from scratch (gpt) + typer et documenter
# - [ ] préciser / nettoyer / compléter le notebook
# - [ ] ajouter régression
# - [ ] ajouter du contenus mathématique, préciser la partie sur Shannon, etc.
# - [ ] ajouter brouillon algos / implémentations Python C4, CART, etc. ?
# - [ ] nettoyer le notebook hors illustrations
# - [ ] mettre au propre les illustrations

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Qu'est-ce qu'un arbre de décision ?
#
# - un modèle non paramétrique (contrairement à un réseau de neurones où on a des paramètres: les poids du réseau)
# - utilisé pour classification et régression
#   - *attributs* (variables) : nominaux, numériques et binaires
#   - *labels* : nominaux, numériques et binaires
# - permet de ... (repréentation sous la forme d'arbre)

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Exemple dataset (classification binaire)
#
# ...
#
# <img src="figs/arbres_decision_dataset_restaurant.png" width="40%" />
#
# voc:
# - *attribut* (= feature, variable)
# - *exemple*
# - *labels* (= sortie, classe)

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Exemple d'arbre (attributs *nominaux*)
#
# ...
#
# voc:
# - *attribut* (= feature, variable)
# - *exemple*
# - *labels* (= sortie, classe)

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Représentation des sous-ensembles (attributs *nominaux*)
#
# <img src="figs/arbres_decision_representation_donnees_nominales.png" width="30%" />
#
# voc:
# - *sous-ensemble*

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Exemple d'arbre (attributs *binaires*)
#
# ...
#
# voc:
# - *attribut* (= feature, variable)
# - *exemple*
# - *labels* (= sortie, classe)

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Représentation des sous-ensembles (attributs *binaires*)
#
# ...
#
# voc:
# - *sous-ensemble*

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Exemple d'arbre (attributs *numériques*)
#
# ...
#
# voc:
# - *attribut* (= feature, variable)
# - *exemple*
# - *labels* (= sortie, classe)

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Représentation des sous-ensembles (attributs *numériques*)
#
# <img src="figs/arbres_decision_representation_donnees_numeriques.png" width="30%" />
#
# <img src="figs/arbres_decision_representation_donnees_numeriques_2.png" width="30%" />
#
# voc:
# - *sous-ensemble*

# %% [markdown] editable=true slideshow={"slide_type": ""} jp-MarkdownHeadingCollapsed=true
# ## Comment construire automatiquement un arbre de décision à partir d'un dataset ?
#
# **Qu'est-ce qu'on veut**
# 1. un arbre qui prédit correctement

# %% [markdown] editable=true slideshow={"slide_type": ""} jp-MarkdownHeadingCollapsed=true
# ## Il y a beaucoup d'abtres qui prédisent correctement les exemples d'un dataset
#
# ...

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Comment construire automatiquement un arbre de décision à partir d'un dataset ?
#
# **Qu'est-ce qu'on veut**
# 1. un arbre qui prédit correctement
# 2. un abre le plus simple possible

# %% [markdown]
# ## Algorithmes naïfs
#
# - brute force
#
# ...

# %% [markdown]
# ## Algorithmes naïfs
#
# - brute force
# - algorithme évolutionniste
#
# ...

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Algorithme glouton
#
# <img src="figs/arbres_decision_algo_commun.jpg" width="30%" />

# %% [markdown]
# ...

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Généralisation : élagage
#
# ...

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Régression
#
# <img src="figs/arbres_decision_regression_representation_donnees_numeriques.png" width="30%" />

# %% [markdown]
#
