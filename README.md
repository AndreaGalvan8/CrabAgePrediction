# CrabAgePrediction
This repository offers an interface to input various physical parameters of a crab to predict their age.



Different Machine Learning algorithms were tested to predict accuracy with data provided in the link below

ML Algorithms Tested:
K nearest neighbors 
RBF SVM
Logistic Regression
Random Forest.

The data:
Sex of crab – Male, Female, or Indeterminate
Length of crab 
Diameter of crab 
Height of crab 
Weight of crab 
Weight of shucked 
Viscera (organs)
Shell weight 
Age of crab

Data source: https://www.kaggle.com/datasets/sidhus/crab-age-prediction

We also only care if a crab is edible or not. We don't care about how old the crab is in months. As such, we will re-classify the data so that we only look at if the crab is old enough to eat or not. In this case, if the crab is older than 11 years the output will be "11 years or younger" and if the crab is less of equal 11 yeas the output would be "Older than 11 years. Therefore, the only parameters needed are sex, lenght, diameter, height, and the whole weight of the crab, that way crabs do not need to be killed in the process. 
