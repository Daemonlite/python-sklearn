
"""
# Data processing with scikit-learn
# starting with data preprocessing

preprocessing methods that we can perform effectively with Scikit-Learn are data encoding and feature scaling
Some of the widely used data encoding methods are Label Encoding and One Hot Encoding.

label-encoding -- Label encoding is basically a way of encoding categorical variables to numerical variables.
Label encoding translates categorical data into numeric form, but it should be applied judiciously.
In contexts like regression models, where numeric values imply ordinality, label encoding can lead to misconceptions.
For instance, assigning numbers to fruits might imply a hierarchy where there isn't one.
thus, to address this, one-hot encoding emerges as an alternative method.

one-hot encoding -- One hot encoding is a method of converting categorical variables into binary variables.


"""
# Data  encoding 

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

basket = ['apple', 'orange', 'grape', 'strawberry', 'melon', 'plum', 'banana', 'melon', 'plum', 'plum', 'grape', 'watermelon', 'melon', 'orange']

encoder = LabelEncoder()
# convert categorical data to numeric
labels = encoder.fit_transform(basket)

#print(labels)

# We can also convert the numerical labels back to the original categorical values by using the function inverse_transform().
revert = encoder.inverse_transform(labels)
#print(revert)

# One hot encoding 
labels2 = encoder.fit_transform(basket).reshape(-1, 1)
# convert categorical data to binary
onehot_encoder = OneHotEncoder()
onehot_labels = onehot_encoder.fit_transform(labels2)
#print(onehot_labels.toarray())


# Feature Scaling
"""
Feature scaling is a method to normalize variables or features of data.
Feature scaling may be necessary in machine learning for several reasons.
It can make the training faster, and it is also capable of making the flow of gradient descent smooth.

sklearns methods are StandardScale and MinMaxScaler using Iris data provided by sklearn

"""

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
#print(iris_df)

#StandardScaler()
#StandardScaler() in Scikit-Learn scales the values so that their mean is 0 and variance is 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
standard_iris = scaler.fit_transform(iris_df)
standard_iris = pd.DataFrame(standard_iris, columns = iris.feature_names)
print("standard iris")
print(standard_iris)
print(standard_iris.mean())
print(standard_iris.var())

# MinMaxScaler
'''
MinMaxScaler() is one method of scaling data, and it converts data into some value in the range [0, 1], 
or else in the ranage [-1, 1] if there are negative values.
 With the iris data, since there are no negative values, the range of data should be between 0 and 1.
'''

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax_iris = minmax.fit_transform(iris_df)
minmax_iris = pd.DataFrame(minmax_iris, columns = iris.feature_names)
print('min max iris')
print(minmax_iris)
print(minmax_iris.min(),minmax_iris.max())

