import pandas as pd
from sklearn.model_selection import train_test_split
import joblib as jb
from treeinterpreter import treeinterpreter as ti
from waterfall_chart import plot as waterfall
import matplotlib.pyplot as plt
import numpy as np

from helpers import ChatGPT_API

data = pd.read_csv('data_complete.csv')

X = data.iloc[:, :-1].values
X = pd.DataFrame(X, columns=data.columns[:-1].values) # make data frame
y = data.iloc[:, -1].values
y = pd.Series(y, name=data.columns[-1])
feature_names = data.columns[:-1].values  # Feature names 17 features
feature_names[12] = "%_Workspace"
feature_names[13] = "Closed_eyes"
print('Feature names:', feature_names)
target_names = ['Sleepy', 'Distracted', 'Focused']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained RandomForest model
model = jb.load('models/random_forest_model.pkl')
instance_index = 6
# Pick a single instance from X_test to interpret
instance = X_test.iloc[instance_index].values.reshape(1, -1)
print('Instance:', instance)

# Get the prediction, bias, and feature contributions using treeinterpreter
prediction, bias, contributions = ti.predict(model, instance)
contributions = np.round(contributions, 3)

predicted_class_index = prediction[0].argmax()  # Index of the predicted class
print('Predicted class:', target_names[predicted_class_index])
print('Prediction probability:', prediction[0][predicted_class_index])
print('Actual class:', target_names[y_test.iloc[instance_index]])
class_contributions = contributions[0][:, predicted_class_index]  # Contributions for all features for the predicted class

# Bias (initial prediction probability before adding contributions)
initial_value = bias[0][predicted_class_index]

# Preparing data for the waterfall chart
# We include the initial_value as the starting point and add the feature contributions
contribution_values = [initial_value] + list(class_contributions)
contribution_labels = ['Bias'] + list(feature_names)
# make dictionary of feature names and contributions
feature_contributions = dict(zip(feature_names, class_contributions))
# sort the dictionary by values based on absolute value
sorted_feature_contributions = dict(sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True))
# remove features with less than absolute value of 0.05
sorted_feature_contributions = {k: v for k, v in sorted_feature_contributions.items() if abs(v) > 0.05}
print('Feature contributions:', sorted_feature_contributions)




# Calculate the variance (or standard deviation) of the tree predictions
# For classification, we will calculate variance for each class's probability
proba_predictions = np.array([tree.predict_proba(instance)[0] for tree in model.estimators_])

# Calculate the mean and standard deviation across all trees
mean_proba = np.mean(proba_predictions, axis=0)
std_proba = np.std(proba_predictions, axis=0)
print('Mean probabilities:', mean_proba)
# The predicted class (based on the highest mean probability)

# Confidence measure: the inverse of standard deviation for the predicted class
confidence = 1 - std_proba[predicted_class_index]

print(f"Predicted class: {target_names[predicted_class_index]}")
print(f"Standard deviation (uncertainty) for predicted class: {std_proba[predicted_class_index]}")
print(f"Confidence: {confidence}")
waterfall(contribution_labels, contribution_values, formatting='{:,.3f}', sorted_value=True, threshold=0.05)
plt.show()


def explain_decision(contributions):
    base = f"""Based on the provided data about the class contribution for the predicted label "Focused", explain which features contributed positively (value > 0) or negatively (value < 0). These values do not represent the actual value of the feature, but rather how much it was taking into account for the prediction. {contributions}"""

    data_description = """
    EAR_Mean: Average eye openness.
    EAR_Std: Variability in eye openness.
    EAR_Median: Median eye openness.
    MAR_Mean: Average mouth openness.
    MAR_Std: Variability in mouth openness.
    MAR_Median: Median mouth openness.
    PUC_Mean: Average pupil roundness.
    PUC_Std: Variability in pupil roundness.
    PUC_Median: Median pupil roundness.
    MOE_Mean: Average combined eye and mouth openness.
    MOE_Std: Variability in combined eye and mouth openness.
    MOE_Median: Median combined eye and mouth openness.
    %_Outside_Workspace: Percentage of time outside workspace boundaries.
    """

    last_data_description = f"""Explain in simple terms what user behavior impacted the decision positively 
    and/ornegatively for the predicted label in less than 50 words.
    Example: The openness of your mouth and fast changing pupil movement contributed the most for prediction 
    "focused", indicating it correspond to foccused behaviour."""

    # Use the ChatGPT API to generate an explanation
    message = base + data_description + last_data_description
    explanation = ChatGPT_API(message)
    print(explanation)

explain_decision(sorted_feature_contributions)

