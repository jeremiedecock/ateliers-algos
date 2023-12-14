import pandas as pd
import numpy as np

from typing import Any, Dict, Optional, Tuple, List

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

# Example: Training the ID3 algorithm
# Let's assume that 'dataset' is a pandas DataFrame containing your data
dataset = pd.read_csv("dataset2.csv")

# The features are the column names of the dataset (except the target feature)
features = dataset.columns[:-1]

target_attribute_name = "play_golf"

# Train the tree
tree = id3_algorithm(dataset, dataset, features, target_attribute_name=target_attribute_name)

# Predict a new instance by calling the 'predict' function
query = dataset.iloc[0,:].to_dict()
query.pop(target_attribute_name)

prediction = predict(query, tree)
print(prediction)
