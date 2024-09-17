import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load and explore dataset
data = pd.read_csv("emails.csv")
print(data.head())
print(data.describe())
print(data.shape)

# Drop unnecessary column and split features and target
data = data.drop('Email No.', axis=1)
X = data.drop(columns=['Prediction'])
y = data['Prediction']

# Scale numeric features
X_numeric = X.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=44)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear', random_state=44),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"{name} Accuracy: {accuracy}")

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Select a model and evaluate
selected_model = models['Random Forest']
evaluate_model(selected_model, X_test, y_test)

# Evaluate all models
for name, model in models.items():
    print(f"{name} Evaluation:")
    evaluate_model(model, X_test, y_test)

# Cross-validation for each model
for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"{name} Cross-Validation Scores: {cv_scores}")
    print(f"{name} Mean Cross-Validation Score: {cv_scores.mean()}")
