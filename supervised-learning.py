# Supervised Learning Algorithm
"""
1. Linear regression
2. Logistic regression
3 . Decision trees
4. Random forests
5. Support Vector Machines

"""

'''
1. Linear regression :
Linear regression is defined as the process of determining the straight line that best fits a set of dispersed data points:
The line can then be projected to forecast fresh data points. 
Because of its simplicity and essential features, linear regression is a fundamental Machine Learning method.

'''

from sklearn.datasets import load_diabetes
import json

diabetes = load_diabetes()

# Convert the dataset to a dictionary
diabetes_dict = {
    "data": diabetes.data.tolist(),
    "target": diabetes.target.tolist(),
    "feature_names": diabetes.feature_names,
    "DESCR": diabetes.DESCR
}

# Save the dictionary to a JSON file
with open("diabetes.json", "w") as file:
    json.dump(diabetes_dict, file)
