# Decision Trees for Classification

## Table of Contents
1. [Introduction to Decision Trees](#introduction-to-decision-trees)
2. [How Decision Trees Work](#how-decision-trees-work)
3. [Mathematical Concepts](#mathematical-concepts)
    - [Entropy](#entropy)
    - [Gini Impurity](#gini-impurity)
    - [Information Gain](#information-gain)
4. [Hyperparameters in Decision Trees](#hyperparameters-in-decision-trees)
5. [Data Structures in Decision Trees](#data-structures-in-decision-trees)
6. [Evaluation Metrics](#evaluation-metrics)
    - [Accuracy](#accuracy)
    - [Precision](#precision)
    - [Recall](#recall)
    - [F1 Score](#f1-score)
    - [Confusion Matrix](#confusion-matrix)
    - [ROC-AUC](#roc-auc)
7. [Practical Examples](#practical-examples)
8. [Interview Questions and Answers](#interview-questions-and-answers)

## Introduction to Decision Trees

A decision tree is a supervised machine learning algorithm used for classification and regression tasks. It splits the data into subsets based on the values of input features, resulting in a tree-like structure of decisions. Each internal node represents a decision based on a feature, each branch represents the outcome of the decision, and each leaf node represents a class label (for classification) or a continuous value (for regression).

## How Decision Trees Work

A decision tree works by recursively splitting the data into subsets based on the feature that provides the highest information gain or the lowest Gini impurity. The process continues until all data is classified, a stopping criterion is met (like maximum depth), or no further splits can be made.

### Real-Time Example
Predicting if a Person Will Buy a Sports Car

Let's say we have the following dataset with features Age, Income, Student, Credit_Rating, and the target variable Buy_Sports_Car.

![Sample_Data](./Images/Sample_Data.png)

## Step-by-Step Implementation of Decision Tree

1. **Calculate Entopy and Information Gain:**
   
   Entropy is a measure of uncertainty or impurity in a dataset. It is calculated using the formula:

   ![Entropy](./Images/Entropy.png)
   
   where p^i  is the proportion of samples that belong to class i.

   For example, ifwehave 9 samples that do not buy a sports car (No) and 5 samples that do (Yes), the entropy is:
   ![Example_Entropy](./Images/Entropy_Example.png)

2. **Splitting the Data:**

We split the data based on different attributes and calculate the weighted average entropy for each split. The attribute with the highest information gain (reduction in entropy) is chosen as the decision node.

Information Gain is calculated as:

## Mathematical Concepts

### Entropy
Entropy measures the uncertainty or impurity in the data. It ranges from 0 (perfect purity) to \(\log_2(k)\) (maximum impurity for k classes).
\[
\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

### Gini Impurity
Gini impurity measures the probability of a randomly chosen element being incorrectly classified. It ranges between 0 (perfect purity) and 0.5 (maximum impurity for binary classification).
\[
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2
\]

### Information Gain
Information gain is the measure of the reduction in entropy or impurity when a dataset is split on an attribute.
\[
\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
\]

## Hyperparameters in Decision Trees

1. **Max Depth:**
   - **Description:** The maximum depth of the tree. Limiting the depth helps prevent overfitting by reducing the complexity of the model.
   - **Example:** `DecisionTreeClassifier(max_depth=5)`

2. **Min Samples Split:**
   - **Description:** The minimum number of samples required to split an internal node. This helps control overfitting by ensuring that nodes are only split when there is enough data.
   - **Example:** `DecisionTreeClassifier(min_samples_split=10)`

3. **Min Samples Leaf:**
   - **Description:** The minimum number of samples required to be at a leaf node. This ensures that leaf nodes have enough data to provide a reliable prediction.
   - **Example:** `DecisionTreeClassifier(min_samples_leaf=5)`

4. **Max Features:**
   - **Description:** The maximum number of features to consider when looking for the best split. This helps in reducing the complexity of the model and can lead to better generalization.
   - **Example:** `DecisionTreeClassifier(max_features='sqrt')`

5. **Criterion:**
   - **Description:** The function to measure the quality of a split. Common criteria are "gini" for Gini impurity and "entropy" for Information Gain.
   - **Example:** `DecisionTreeClassifier(criterion='entropy')`

6. **Max Leaf Nodes:**
   - **Description:** Limits the number of leaf nodes in the decision tree. This parameter helps in controlling the size of the tree and can prevent overfitting.
   - **Example:** `DecisionTreeClassifier(max_leaf_nodes=20)`

7. **Min Impurity Decrease:**
   - **Description:** A node will be split only if this split induces a decrease of the impurity greater than or equal to this value. It helps in making the tree simpler by avoiding splits that do not significantly reduce impurity.
   - **Example:** `DecisionTreeClassifier(min_impurity_decrease=0.01)`

## Data Structures in Decision Trees

1. **Arrays:**
   - Used to store datasets and attributes. Example: `numpy` arrays for feature and label storage.

2. **Linked Lists (Tree Structures):**
   - Nodes are linked to form a tree structure. Example: Each `DecisionTreeNode` object points to its child nodes.

3. **Stacks:**
   - Used for Depth-First Search (DFS) traversal. Example: A stack to keep track of nodes during DFS.

4. **Queues:**
   - Used for Breadth-First Search (BFS) traversal. Example: A queue to keep track of nodes during BFS.

## Evaluation Metrics

### Accuracy
The proportion of true results (both true positives and true negatives) among the total number of cases examined.
\[
\text{Accuracy} = \frac{\text{True Positives + True Negatives}}{\text{Total Samples}}
\]

### Precision
The proportion of true positive results in the positive predicted cases. It indicates the accuracy of the positive predictions.
\[
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
\]

### Recall
The proportion of true positive results out of all actual positive cases. It indicates the model's ability to detect positive cases.
\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
\]

### F1 Score
The harmonic mean of precision and recall. It balances the two metrics and is useful for imbalanced datasets.
\[
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}
\]

### Confusion Matrix
A table used to describe the performance of a classification model, showing the actual versus predicted classifications.
\[
\begin{array}{cc|c|c}
& & \text{Predicted Negative} & \text{Predicted Positive} \\
\hline
& \text{Actual Negative} & \text{True Negative (TN)} & \text{False Positive (FP)} \\
\hline
& \text{Actual Positive} & \text{False Negative (FN)} & \text{True Positive (TP)} \\
\end{array}
\]

### ROC-AUC
Measures the ability of a classifier to distinguish between classes. The ROC curve is a plot of the true positive rate against the false positive rate.
\[
\text{AUC} = \frac{\text{True Positive Rate}}{\text{False Positive Rate}}
\]

## Practical Examples

### Hyperparameter Tuning Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
dt = DecisionTreeClassifier()

# Define the hyperparameters grid
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [None, 10, 20],
    'min_impurity_decrease': [0.0, 0.01, 0.1]
}

# Perform grid search
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
print("Model accuracy on test set:", best_model.score(X_test, y_test))
