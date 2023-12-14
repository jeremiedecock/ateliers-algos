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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Decision trees and random forests

# %% editable=true slideshow={"slide_type": "skip"}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from typing import Any, Dict, Optional, Tuple, List

from IPython.display import Image   # To display graphviz images in the notebook

# %% editable=true slideshow={"slide_type": "skip"}
sns.set_context("talk")

# %% [markdown] editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true
# ## What are decision trees?
#
# - Non-parametric supervised learning methods
# - For both classification and regression tasks
#   - *Attributes* (a.k.a features or variables): can be categorical, numerical or binary
#   - *Labels* (a.k.a outputs): can be categorical, numerical or binary
# - Learn simple if-then-else decision rules that can be represented on a tree

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## What are decision trees?
#
# Tree Structure:
# - Represents a hierarchy of decisions
# - Each internal node denotes a test on an *attribute*
# - Each branch represents an outcome of the test
# - Leaf nodes hold the final decision or prediction
#
# A tree can be seen as a piecewise constant approximation

# %% editable=true slideshow={"slide_type": "skip"}
# !dot -Tsvg tree2.dot > figs/tree2.svg

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Example (here for binary classification with categorical attributes)
#
# <img src="figs/tree2.svg" width="60%" />
#
# **Voc**:
#
# - *Attributes* (features or variables)
# - Outcome
# - Examples
# - Labels (outputs, classes)

# %% editable=true slideshow={"slide_type": "skip"}
#Image('tree1.png')

# %% editable=true slideshow={"slide_type": "slide"}
pd.read_csv("dataset_golf_1.csv")


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## How to automatically build a decision tree from a dataset?
#
# **What do we want**
# 1. A tree that accurately predicts

# %% [markdown] editable=true slideshow={"slide_type": "fragment"} jp-MarkdownHeadingCollapsed=true
# There are many trees that predict examples from a dataset with equivalent accuracy

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## How to automatically build a decision tree from a dataset?
#
# **What do we want**
# 1. A tree that accurately predicts
# 2. *A tree as simple as possible*

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Naive algorithms
#
# - Brute force
#
# We could proceed by brutforce, testing all possible trees for a given set of attributes and measuring their size and accuracy, but this is not feasible in practice: the number of possible trees grows exponentially with the number of attributes

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Naive algorithms
#
# - Brute force
# - Evolutionary algorithms
#
# We could use evolutionary algos, but here too we'll quickly be limited.

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Greedy algorithms
#
# In practice, greedy method is recursively used to build a decision tree from a dataset:

# %% [markdown] editable=true slideshow={"slide_type": "fragment"}
# 1. **Selection of the Best Attribute**: At each step in the algorithm, ID3 chooses the attribute that is most useful for classifying the data. This is done using a measure like Information Gain or Gain Ratio. The attribute with the highest Information Gain (or another chosen metric) is selected as the decision node.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"}
# 2. **Tree Construction**:
#   - Start with all the training instances and a set of all the attributes.
#   - Choose the best attribute using a greedy strategy (highest Information Gain, for example).
#   - Make that attribute a decision node and divide the dataset into smaller subsets based upon the values of this attribute.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"}
# 3. **Recursive Splitting**:
#   - For each subset of data (which is now smaller than the original set):
#     - If all instances in the subset belong to the same class or there are no more attributes to be selected, then create a leaf node with the class label.
#     - If there are mixed instances, then repeat the process: choose the best attribute for this subset of data and split it further. This is the recursive part, where the algorithm repeats the process of attribute selection and tree construction for each new subset.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"}
# 4. **Termination**: The recursion is terminated when either all instances at a node belong to the same class, there are no more attributes left to split upon, or the tree reaches a predefined depth limit.

# %% [markdown] editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true
# Recursive construction of a decision tree
#
# **PROCEDURE** Build-tree (node X)
# <br>
# **IF** All points in X belong to the same class then <br>
# $\quad$ Create a leaf bearing the name of this class <br>
# **ELSE** <br>
# $\quad$ Choose the best attribute to create a node <br>
# $\quad$ The test associated with this node splits $X$ into two parts, denoted $X_g$ and $X_d$ <br>
# $\quad$ *Build-tree*($X_g$) <br>
# $\quad$ *Build-tree*($X_d$) <br>

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## How do you "choose the best attribute to create a node"?
#
# - Entropy
# - Information gain

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Entropy [TODO]
#
# Shannon... théorie de l'info...
#
# Entropy = "how much info you are missing"
#
# e.g. on veut une adresse,
# - France = ... -> there is a lot of info missing
# - 92012 = ... -> less information mission -> lower entropy
#
# In other words : moins nous avons d'entropie plus nous en savons (on cherche a minimiser l'entropie)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# entropie : mesurée en bit
#
# entropie = $-p_{(+)} \log_2 p_{(+)} - p_{(-)} \log_2 p_{(-)}$
#
# e.g.
# - subset 1 = 0 bit
# - subset 2 = 1 bit

# %% [markdown]
# Exercice: calculer l'entropie des sous-ensembles suivants : ...

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Information gain [TODO]
#
# Entropy = "how pure a subset is"
# However it doesn't actually help the algo to choose the attribute
# (...)
#
# Information gain (..)
#
# information gain = entropie(parent) - weighted avg(entropie des sous ensembles)
#
# We want to maximize the information gain
#
# Ex: ...

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Dérouler l'algo complètement

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Ce qu'on a vu : ID3

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## How / when does the algorithm stop splitting ?
#
# There are three options for selecting when a decision tree algorithm stops splitting:
#
# 1. Allow the tree to split until every subset is pure. This means that, if necessary, the algorithm will keep splitting until each end node (leaf) subset contains 1 example and is therefore 100% pure. This might seem desirable, but it can lead to a problem known as overfitting, which we will cover in the next chapter.
# 2. Stop the tree from splitting until every leaf subset is pure. This might seem like a good option, but it can quickly lead to a high error rate and poor performance because the tree is not robust enough.
# 3. Adopt a stopping method. This is the when to stop splitting used by decision tree algorithms to determine when to stop splitting.

# %% [markdown] editable=true slideshow={"slide_type": "fragment"}
# Depending on the type of tree you are using, there are multiple approaches to choose from. Some of these include:
#
# - Stopping when a tree reaches a maximum number of levels, or depth.
# - Stopping when a minimum information-gain level is reached.
# - Stopping when a subset contains less than a defined number of data points.

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Python implementation

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Implement the entropy function

# %%
def entropy(target_col: np.ndarray) -> float:
    """
    Calculate the entropy of a dataset.

    Parameters
    ----------
    target_col : np.ndarray
        The target column

    Returns
    -------
    float
        The entropy of the dataset

    """
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Test the entropy function

# %%
df = pd.read_csv("dataset_golf_1.csv", dtype=str)
df

# %% [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### On all labels

# %%
labels = df.label.values
print(f"Entropy of subset {labels.tolist()} = {entropy(labels)}")

# %% [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### On a subset of labels

# %%
labels = df.label.values[:2]
print(f"Entropy of subset {labels.tolist()} = {entropy(labels)}")

# %% [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### On another subset of labels

# %%
labels = df.label.values[:4]
print(f"Entropy of subset {labels.tolist()} = {entropy(labels)}")


# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Implement the info_gain function

# %%
def info_gain(data: pd.DataFrame, split_attribute_name: str, target_name: Optional[str] = "class") -> float:
    """
    Calculate the information gain of a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset for whose feature the IG should be calculated
    split_attribute_name : str
        The name of the feature for which the information gain should be calculated
    target_name : str, optional
        The name of the target feature. The default is "class"

    Returns
    -------
    float
        The information gain of the dataset
    """
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    # Calculate the values and the corresponding counts for the split attribute 
    vals, counts= np.unique(data[split_attribute_name], return_counts=True)
    
    # Calculate the weighted entropy
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    # Calculate the information gain
    information_gain = total_entropy - weighted_entropy

    return information_gain


# %% [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Test the info_gain function

# %%
info_gain(df, "outlook", target_name="label")

# %%
info_gain(df, "humidity", target_name="label")

# %%
info_gain(df, "wind", target_name="label")

# %%
info_gain(df, "temperature", target_name="label")


# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Implement the id3_algorithm function

# %%
def id3_algorithm(data: pd.DataFrame, original_data: pd.DataFrame, features: List[str], target_attribute_name: str = "class", parent_node_class: Optional[Any] = None) -> Any:
    """
    ID3 Algorithm. This function takes five parameters:

    Parameters
    ----------
    data : pd.DataFrame
        The data for which the ID3 algorithm should be applied
    original_data : pd.DataFrame
        This is the original dataset needed to calculate the mode target feature value of the original dataset in the case the dataset delivered by the first parameter is empty
    features : List[str]
        The feature space of the dataset. This is needed for the recursive call since during the tree growing process
    target_attribute_name : str, optional
        The name of the target attribute. The default is "class"
    parent_node_class : Any, optional
        This is the value or class of the mode target feature value of the parent node for a specific node. This is also needed for the recursive call in the case the dataset is empty. The default is None

    Returns
    -------
    Any
        The prediction result
    """
    # Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
    
    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.
    elif len(features) == 0:
        return parent_node_class
    
    # If none of the above holds true, grow the tree!
    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Select the feature which best splits the dataset
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features] # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information gain in the first run
        tree = {best_feature:{}}
        
        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        # Grow a branch under the root node for each possible value of the root node feature
        
        for value in np.unique(data[best_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest information gain and therewith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            
            # Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = id3_algorithm(sub_data, dataset, features, target_attribute_name, parent_node_class)
            
            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)


# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Test the id3_algorithm function

# %%
# 'dataset' is a pandas DataFrame containing your dataset
dataset = pd.read_csv("dataset_golf_1.csv")

# The features (attributes) are the column names of the dataset (except the target feature)
features = dataset.columns[:-1]

# The target
target_attribute_name = "label"

# Train the tree
decision_tree = id3_algorithm(dataset, dataset, features, target_attribute_name=target_attribute_name)

# %% editable=true slideshow={"slide_type": ""}
decision_tree

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Implement a function to plot the tree (WIP 😅)

# %% editable=true slideshow={"slide_type": ""}
from graphviz import Digraph

def add_nodes_edges(tree, parent_name, graph):
    if isinstance(tree, dict):
        for node, subtree in tree.items():
            graph.node(node)
            if parent_name is not None:
                graph.edge(parent_name, node)
            add_nodes_edges(subtree, node, graph)
    else:
        # Leaf node
        graph.node(tree)
        if parent_name is not None:
            graph.edge(parent_name, tree)

def tree_to_dot(tree):
    graph = Digraph()
    add_nodes_edges(tree, None, graph)
    return graph


# %%
dot = tree_to_dot(decision_tree)
dot.render('decision_tree.dot', format='svg')


# %% [markdown] editable=true slideshow={"slide_type": "skip"}
# <img src="decision_tree.dot.svg" width="30%" />

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Implement the predict function

# %%
def predict(query: Dict[str, Any], tree: Dict[str, Any], default: Optional[int] = 1) -> Any:
    """
    Prediction of a new/unseen query instance. This takes three parameters:

    Parameters
    ----------
    query : Dict[str, Any]
        A dictionary of the shape {"feature_name":feature_value,...}
    tree : Dict[str, Any]
        The tree that was trained on the training data (ID3)
    default : int, optional
        The prediction that will be returned if the query instance is not applicable in the tree. Default is 1.

    Returns
    -------
    Any
        The prediction result
    """
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result


# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Test the predict function

# %%
# Predict a new instance by calling the 'predict' function
query = dataset.iloc[0,:].to_dict()
query.pop(target_attribute_name)

prediction = predict(query, tree)
print(prediction)

# %% [markdown] jp-MarkdownHeadingCollapsed=true editable=true slideshow={"slide_type": "slide"}
# ## Other selection criteria
# ### Gini Impurity
#
# $$
# Gini(T) = 1 - \sum_{i=1}^k p_i^2
# $$
#
# - T represent a training dataset
# - p is the probability of "T" belonging to class "i"
#
# The lower the impurity the better.

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Decision Tree Algorithms Overview

# %% [markdown] jp-MarkdownHeadingCollapsed=true editable=true slideshow={"slide_type": "slide"}
# - **ID3 (Iterative Dichotomiser 3)**
#   - **entropy** and **information gain**
#   - Developed in 1979 by Ross Quinlan.
#   - Creates a multiway tree.
#   - Selects categorical features at each node for the largest information gain.
#   - Trees grown to maximum size, then pruned for better generalization to new data.

# %% [markdown] jp-MarkdownHeadingCollapsed=true editable=true slideshow={"slide_type": "slide"}
# - **C4.5 (Successor to ID3)**
#   - **entropy** and **gain ratio**
#   - Developed in 1986 by Ross Quinlan.
#   - Handles both categorical and continuous features.
#   - Converts decision trees into if-then rules.
#   - Rules are ordered based on accuracy.
#   - Prunes rules by evaluating accuracy improvements.

# %% [markdown] jp-MarkdownHeadingCollapsed=true editable=true slideshow={"slide_type": "slide"}
# - **C5.0 (Latest Version by Quinlan)**
#   - Proprietary license.
#   - More efficient in memory and rule size than C4.5.
#   - Higher accuracy.

# %% [markdown] jp-MarkdownHeadingCollapsed=true editable=true slideshow={"slide_type": "slide"}
# - **CART (Classification and Regression Trees)**
#   - **gini impurty** or **variance reduction**
#   - Developed in 1984 by Brieman, Friedman, Ohlson and Stone
#   - Similar to C4.5, but supports numerical targets for regression.
#   - Builds binary trees based on largest information gain.
#   - Does not generate rule sets.

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Potential problems with decision trees
#
# - Overfitting
# - ...

# %% editable=true slideshow={"slide_type": "skip"}
pd.read_csv("dataset_golf_1.csv")

# %% editable=true slideshow={"slide_type": "skip"}
pd.read_csv("dataset_golf_2.csv")

# %% editable=true slideshow={"slide_type": "skip"}
dataset = pd.read_csv("dataset_golf_1.csv")
features = dataset.columns[:-1]
target_attribute_name = "label"
decision_tree = id3_algorithm(dataset, dataset, features, target_attribute_name=target_attribute_name)

decision_tree

# %% editable=true slideshow={"slide_type": "slide"}
dot = tree_to_dot(decision_tree)
dot.render('decision_tree1.dot', format='svg')

# %% [markdown] editable=true slideshow={"slide_type": "skip"}
# <img src="decision_tree1.dot.svg" width="30%" />

# %% editable=true slideshow={"slide_type": "skip"}
dataset = pd.read_csv("dataset_golf_2.csv")
features = dataset.columns[:-1]
target_attribute_name = "label"
decision_tree = id3_algorithm(dataset, dataset, features, target_attribute_name=target_attribute_name)

decision_tree

# %% editable=true slideshow={"slide_type": "slide"}
dot = tree_to_dot(decision_tree)
dot.render('decision_tree2.dot', format='svg')

# %% [markdown] editable=true slideshow={"slide_type": "skip"}
# <img src="decision_tree2.dot.svg" width="30%" />

# %% [markdown] editable=true slideshow={"slide_type": "slide"} jp-MarkdownHeadingCollapsed=true
# ## Generalization
#
# - Statistical sifificance tests
# - Pruning

# %% [markdown] editable=true slideshow={"slide_type": "skip"}
# ## Regression [TODO]
#
# <img src="figs/arbres_decision_regression_representation_donnees_numeriques.png" width="30%" />

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Scikit-Learn implementation
#
# ### Models
#
# - **Classification** `sklearn.tree.DecisionTreeClassifier` -> [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
# - **Regression** `sklearn.tree.DecisionTreeRegressor` -> [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)
#
# ### Algorithms
#
# Scikit-Learn uses an optimized version of the CART algorithm
#
# See also: https://scikit-learn.org/stable/modules/tree.html

# %% [markdown] jp-MarkdownHeadingCollapsed=true editable=true slideshow={"slide_type": "slide"}
# ### Warning
#
# The Scikit-Learn implementation does not support categorical variables for now!
#
# See: https://scikit-learn.org/stable/modules/tree.html

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Classification: Example with the Iris dataset
#
# - **Number of Instances**: 150 (50 in each of three classes)
# - **Number of Attributes**: 4 numeric, predictive attributes and the class
# - **Attribute Information**:
#   - *sepal length* in cm (min: 4.3, max: 7.9)
#   - *sepal width* in cm  (min: 2.0, max: 4.4)
#   - *petal length* in cm (min: 1.0, max: 6.9)
#   - *petal width* in cm  (min: 0.1, max: 2.5)
# - **class**:
#   - Iris-Setosa (33.3% of the dataset)
#   - Iris-Versicolour (33.3% of the dataset)
#   - Iris-Virginica (33.3% of the dataset)
#
# (see [doc1](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris) and [doc2](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset))

# %% editable=true slideshow={"slide_type": "skip"}
from sklearn.datasets import load_iris
from sklearn import tree

dataset = load_iris()
X, y = dataset.data, dataset.target

# %% editable=true slideshow={"slide_type": "skip"}
pd.DataFrame(X).hist();

# %% editable=true slideshow={"slide_type": "skip"}
pd.DataFrame(y).hist();

# %% editable=true slideshow={"slide_type": "skip"}
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# %% editable=true slideshow={"slide_type": "skip"}
tree.plot_tree(clf);

# %% editable=true slideshow={"slide_type": "skip"}
clf.predict([[5.84, 3.05, 3.76, 1.20]])

# %% editable=true slideshow={"slide_type": "skip"}
import graphviz    # !pip install graphviz

# %% editable=true slideshow={"slide_type": "skip"}
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=dataset.feature_names,  
                                class_names=dataset.target_names,  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# %% editable=true slideshow={"slide_type": "skip"}
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = dataset.data[:, pair]
    y = dataset.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=dataset.feature_names[pair[0]],
        ylabel=dataset.feature_names[pair[1]],
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=dataset.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0);
#_ = plt.axis("tight");

# %% [markdown]
# ### Regression
#

# %%
from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])

# %%
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_2 = tree.DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Advantages of Decision Trees
#
# - **Easy to Understand and Interpret**: Trees can be visualized, making them easy to understand and interpret, even for non-technical stakeholders.
# - **Handles Both Numerical and Categorical Data**: They can handle datasets that have both numerical and categorical variables.
# - **No Need for Data Preprocessing**: Often requires little data preparation. They do not require normalization of data or dummy variables.
# - **Non-Parametric Method**: Since they are non-parametric, they are not constrained by a particular distribution of data.
# - **Automatic Feature Selection**: Decision trees implicitly perform feature selection during training, which is beneficial in cases with a large number of features- Can Model Non-Linear Relationships: Effective at capturing non-linear relationships between features and labels.
# - **Useful for Exploratory Analysis**: Can be used to identify the most influential variables in a dataset.

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Disadvantages of Decision Trees
#
# - **Overfitting**: Tend to overfit the data, especially if the tree is deep with many branches. This can be mitigated using techniques like pruning, setting a maximum depth, or minimum samples per leaf.
# - **Variance**: Small variations in the data can result in a completely different tree. This can be reduced by using ensemble methods like Random Forests.
# - **Not Ideal for Continuous Variables**: They are not the best choice for continuous numerical data, as they lose information when categorizing variables into different nodes.
# - **Biased with Imbalanced Data**: Decision trees can be biased towards dominant classes, so they might require balancing before being used.
# - **Instability**: A small change in the data can lead to a significant change in the structure of the decision tree, making them quite unstable.
# - **Greedy Algorithms**: Decision trees use a greedy approach which might not result in the globally optimal tree.
# - **Difficulty in Capturing Complex Relationships**: They may struggle to capture more complex relationships without becoming overly complex themselves.

# %% [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Bibliography
#
# - R. Quinlan, "Learning efficient classification procedures", Machine Learning: an artificial intelligence approach, Michalski, Carbonell & Mitchell (eds.), Morgan Kaufmann, 1983, p. 463-482.
# - R. Quinlan, "The effect of noise on concept learning", Machine Learning: an artificial intelligence approach, Vol. II, Michalski, Carbonell & Mitchell (eds.), Morgan Kaufmann, 1986, p. 149-166.
# - R. Quinlan, "Induction of decision trees", Machine learning, 1 (1), p. 81-106, Kluwer.
# - J. Cheng, U. Fayyad, K. Irani, Z. Quian, "Improved decision trees: a generalized version of ID3", International ML Conference, 1988, Ann-Arbor, p. 100-106.
# - P. Utgoff, "ID5: an incremental ID3", International ML Conference, 1988, Ann-Arbor, p. 107-120.
# - J. Wirth, J. Catlett, "Experiments on the Costs and Benefits of Windowing in ID3", International ML Conference, 1988, Ann-Arbor, p. 87-99.
# - Breiman, Leo. "Bagging predictors." Machine learning 24 (1996): 123-140. [PDF](https://link.springer.com/content/pdf/10.1007/BF00058655.pdf)
# - Breiman, Leo. "Random forests." Machine learning 45 (2001): 5-32. [PDF](https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf)
# - Breiman, Leo. Classification and regression trees. Routledge, 2017.
# - Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785-794. 2016. [PDF](https://dl.acm.org/doi/pdf/10.1145/2939672.2939785)
# - Friedman, Jerome H. "Greedy function approximation: a gradient boosting machine." Annals of statistics (2001): 1189-1232. [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.pdf)
# - Mason, Llew, Jonathan Baxter, Peter Bartlett, and Marcus Frean. "Boosting algorithms as gradient descent." Advances in neural information processing systems 12 (1999). [PDF](https://proceedings.neurips.cc/paper_files/paper/1999/file/96a93ba89a5b5c6c226e49b88973f46e-Paper.pdf)
