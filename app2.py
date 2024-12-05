import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("CrabAgePrediction.csv") 

# Preprocess the data
data['Age'] = data['Age'].apply(lambda x: 1 if x > 11 else 0)
data['Sex'] = data['Sex'].map({'F': 0, 'M': 1, 'I': 2})

data = data.drop(columns=["Shucked Weight", "Viscera Weight", "Shell Weight"])

#  target
X = data.drop(["Age"], axis=1)
y = data["Age"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RBF SVM
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Streamlit App Interface
st.title("Crab Age Prediction")
st.header("Enter Crab Attributes for Prediction")
st.subheader("Nicholas Jurczyk, Brandt Sandman, Andrea Galvan")

# Input fields for the user, As we cannot measure the Viscera (organ) weight, shell weight, nor shucked weight without killing the crab, we will remove these from the dataset
sex = st.selectbox("Sex (Female=0, Male=1, Indeterminate=2)", options=[0, 1, 2])
length = st.number_input("Length (mm)", min_value=0.0, step=0.1)
diameter = st.number_input("Diameter (mm)", min_value=0.0, step=0.1)
height = st.number_input("Height (mm)", min_value=0.0, step=0.1)
whole_weight = st.number_input("Whole Weight (g)", min_value=0.0, step=0.1)


# predict the age
if st.button("Predict Age"):
    user_input = np.array([[sex, length, diameter, height, whole_weight]])
    prediction = log_reg_model.predict.predict(user_input)
    result = "Older than 11 years" if prediction[0] == 1 else "11 years or younger"
    st.success(f"The predicted age of the crab is: {result}")

# model's accuracy
accuracy = log_reg_model.score(X_test, y_test)
st.write(f"Model Accuracy (Using Logistic Rgression): {accuracy * 100:.2f}%")
