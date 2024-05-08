import pandas as pd

# Read the dataset
df = pd.read_csv('heart_disease_uci.csv')

# Display the first few rows of the dataset to ensure it was loaded correctly
print(df.head())

from sklearn.model_selection import train_test_split

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['target [0=no heart disease; 1 = heart disease ]'])
y = df['target [0=no heart disease; 1 = heart disease ]']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Display basic statistics of numerical features
print("Basic statistics of numerical features:")
print(df.describe())

# Visualize distributions of numerical features
import matplotlib.pyplot as plt

# Visualize distributions of numerical features in a single figure with color
num_cols = df.columns[:-1]  
num_rows = (len(num_cols) + 1) // 2  

plt.figure(figsize=(15, 10))

# Define colors for each subplot
colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'purple', 'cyan', 'pink', 'yellow']

for i, column in enumerate(num_cols):
    plt.subplot(num_rows, 2, i + 1)
    plt.hist(df[column], bins=20, color=colors[i % len(colors)], edgecolor='black')
    plt.title(f'Distribution of {column}', fontsize=10, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.show()

import tensorflow as tf

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the testing dataset
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Testing Loss: {loss:.4f}')
print(f'Testing Accuracy: {accuracy:.4f}')

from sklearn.metrics import classification_report

# Predict probabilities on the testing set
y_pred_prob = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred = (y_pred_prob > 0.5).astype(int)

# Generate classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# Define thresholds for risk categories
low_threshold = 0.3
high_threshold = 0.7

# Categorize patients based on predicted probabilities
def categorize_patients(predictions, low_threshold, high_threshold):
    categories = []
    for pred in predictions:
        if pred < low_threshold:
            categories.append("Low Risk")
        elif pred >= high_threshold:
            categories.append("High Risk")
        else:
            categories.append("Medium Risk")
    return categories

# Categorize patients in the testing set
risk_categories = categorize_patients(y_pred_prob, low_threshold, high_threshold)

# Print the distribution of risk categories
print("Distribution of Risk Categories:")
print(pd.Series(risk_categories).value_counts())

# Define hypothetical treatment strategies
treatment_recommendations = {
    "Low Risk": "Regular check-ups and lifestyle modification (e.g., diet, exercise)",
    "Medium Risk": "Regular check-ups and medication (if necessary) in addition to lifestyle modification",
    "High Risk": "Immediate medical intervention and aggressive treatment"
}

# Simulate treatment recommendations for each patient
def simulate_treatment_recommendations(patients, risk_categories, recommendations):
    for patient, category in zip(patients, risk_categories):
        print(f"Patient {patient}: {recommendations[category]}")

# Simulate treatment recommendations for patients in the testing set
simulate_treatment_recommendations(range(1, len(y_test) + 1), risk_categories, treatment_recommendations)

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of target classes
plt.figure(figsize=(8, 6))
sns.countplot(x='target [0=no heart disease; 1 = heart disease ]', data=df, palette='Set2')
plt.title('Distribution of Target Classes')
plt.xlabel('Cases (1:Heart disease, 0:Normal)')
plt.ylabel('Count')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Convert feature importances to percentage
total_importance = sum(feature_importances)
feature_importances_percent = (feature_importances / total_importance) * 100

# Manually specify x-axis labels
x_labels = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 
            'Cholesterol', 'Max Heart Rate', 'ST Depression', 'Num Major Vessels']

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances_percent)
plt.xticks(range(len(feature_importances)), x_labels, rotation=45, fontsize=14)  # Adjust the font size and rotation angle here
plt.title('Feature Importance in Predicting Heart Disease', fontname='Times New Roman', fontweight='bold', fontsize=14)
plt.xlabel('Features', fontname='Times New Roman')
plt.ylabel('Importance (%)', fontname='Times New Roman', fontsize=14)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data
scaler.fit(X_train)

# Function to preprocess the data for a new patient
def preprocess_new_patient_data(new_patient_data, scaler):
    # Convert the input dictionary to a DataFrame with appropriate column names
    new_patient_df = pd.DataFrame([new_patient_data])
    
    # Scale the numerical features using the provided scaler
    scaled_features = scaler.transform(new_patient_df)
    
    return scaled_features

# Assuming `model` is a trained model (e.g., RandomForestClassifier or Sequential)

# Assuming `scaler` is a fitted StandardScaler object

# Collect input data for the new patient
new_patient_data = {
    'age in years': 75,
    'gender (Male = 1, Female = 2)': 2,  # Assuming 1 for male and 2 for female
    'chest pain type (typical angina=1, asymptomatic=2, non-anginal=3, atypical angina =4)': 4,  
    'resting blood pressure': 190,
    'cholesterol measure': 240,
    'Maximum heart rate achieved': 90,
    'ST depression induced by exercise relative to rest': 2.5,
    'number of major vessels (0-3) colored by flourosopy': 3
}

# Predict the probability of heart disease for the new patient
preprocessed_data = preprocess_new_patient_data(new_patient_data, scaler)


# Predict the probability of heart disease for the new patient
def predict_probability_for_new_patient(new_patient_data, model, scaler):
    # Preprocess the data for the new patient
    preprocessed_data = preprocess_new_patient_data(new_patient_data, scaler)
    
    # Predict the probability of heart disease using the trained model
    probability = model.predict(preprocessed_data)  # Prediction for each class
    return probability

# Predict the probability of heart disease for the new patient
predicted_probability = predict_probability_for_new_patient(new_patient_data, model, scaler)
print("Predicted Probability of Heart Disease:", predicted_probability)

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Plotting classification report
def plot_classification_report(classification_rep):
    # Splitting the classification report into lines
    lines = classification_rep.split('\n')

    # Extracting class names and metrics
    classes = []
    plotMat = []
    for line in lines[2:]:  
        t = line.strip().split()
        if len(t) == 0:
            break
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    # Converting metrics into an array and plotting
    plotMat = np.array(plotMat)
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, plotMat[i, j], ha='center', va='center', color='white' if plotMat[i, j] > 0.5 else 'black')
    plt.imshow(plotMat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Classification Report')
    plt.colorbar()
    x_tick_marks = np.arange(len(classes))
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, classes, rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.tight_layout()
    plt.show()

# Generate classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot ROC curve
plot_roc_curve(y_test, y_pred_prob)

# Plot Classification Report
plot_classification_report(classification_rep)


import pandas as pd

# Write feature importances to a CSV file
feature_importance_df = pd.DataFrame({'Feature': x_labels, 'Importance (%)': feature_importances_percent})
feature_importance_df.to_csv('feature_importance_heart_disease.csv', index=False)
print("Feature importances saved to 'feature_importance_heart_disease.csv'")