import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
sns.set()

# Fetch the file
original_data = pd.read_csv('CrabAgePrediction.csv') # Local, use full path if notebook and file in different folders! 

#Display What attributes we are working with in the Data
original_data.head()

original_data.describe(include='all') # descriptive statistics for all columns

original_data.isnull().sum() # check for null values

original_data[original_data.duplicated(keep=False)] # check for duplicate rows

#for the data, we need to turn the Label into a number so we can compare if we predicted the right thing.
# Step 1: Create a mapping dictionary
mapping = {
    'F': 0,
    'M': 1,
    'I': 2
}

# Step 2: Apply the mapping to the specific column
original_data['Sex'] = original_data['Sex'].map(mapping)

original_data.head()

# Code source: ChatGPT
# Remap the 'age' column where age > 12 becomes 1 and age <= 12 becomes 0
original_data['Age'] = original_data['Age'].apply(lambda x: 1 if x > 11 else 0)

original_data.head()

target = original_data["Age"]
predictors = original_data.drop(["Age"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    predictors, target, test_size=0.2, random_state=42
)  # 80-20 split into training and test data

# Check data balancing
y_train.value_counts()

# There is no severe skew in the class distribution. No resampling needed.
# If you want to learn more about resampling, also check https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/

#Create a Heat Map to better understand 
f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(original_data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()

# Code from ChatGPT

# Step 1: Initialize the KNN model, specify number of neighbors (k)
# Tests conducted between 1-15. 8 yielded the best results.
k = 8 # You can experiment with different values of k
knn = KNeighborsClassifier(n_neighbors=k)

# Step 2: Train the KNN model on the training dataset
knn.fit(X_train, y_train)

# Step 3: Use the trained model to make predictions on the test dataset
y_pred = knn.predict(X_test)

# Step 4: Evaluate the model

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Import necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Initialize the RBF SVM model
rbf_svm = SVC(kernel='rbf')

# Step 2: Train the RBF SVM model on the training data
rbf_svm.fit(X_train, y_train)

# Step 3: Make predictions using the trained model on the test data
y_pred = rbf_svm.predict(X_test)

# Step 4: Evaluate the model

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Code from ChatGPT

# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Initialize the Logistic Regression model
log_reg_model = LogisticRegression()

# Step 2: Train the model on the training data
log_reg_model.fit(X_train, y_train)

# Step 3: Make predictions using the trained model on the test data
y_pred = log_reg_model.predict(X_test)

# Step 4: Evaluate the model

# 4.1. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Code from ChatGPT

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Initialize the Random Forest model
# n_estimators is the number of trees in the forest (can be tuned)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 2: Train the model on the training data
rf_model.fit(X_train, y_train)

# Step 3: Make predictions using the trained model on the test data
y_pred = rf_model.predict(X_test)

# Step 4: Evaluate the model

# 4.1. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
