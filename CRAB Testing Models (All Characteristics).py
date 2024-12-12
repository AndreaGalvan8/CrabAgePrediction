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

