import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
num_cols = df.columns[:-1]  # Exclude the target variable
num_rows = (len(num_cols) + 1) // 2  # Calculate the number of rows for subplots

plt.figure(figsize=(15, 10))

# Define colors for each subplot
colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'purple', 'cyan', 'pink', 'yellow']

for i, column in enumerate(num_cols):
    plt.subplot(num_rows, 2, i + 1)
    plt.hist(df[column], bins=20, color=colors[i % len(colors)], edgecolor='black')
    plt.xlabel(column, fontsize=12, fontweight='bold')  # Increase font size and make xlabel bold
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')  # Increase font size and make ylabel bold

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

# Plotting the distribution of target classes
plt.figure(figsize=(8, 6))
sns.countplot(x='target [0=no heart disease; 1 = heart disease ]', data=df, palette='Set2')
plt.title('Distribution of Target Classes', fontsize=14, fontweight='bold')  # Make title bold and larger
plt.xlabel('Cases (1:Heart disease, 0:Normal)', fontsize=12, fontweight='bold')  # Increase font size and make xlabel bold
plt.ylabel('Count', fontsize=12, fontweight='bold')  # Increase font size and make ylabel bold
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
plt.xticks(range(len(feature_importances)), x_labels, rotation=45, fontsize=14, fontweight='bold')  # Adjust the font size, make labels bold and rotate them
plt.title('Feature Importance in Predicting Heart Disease', fontweight='bold', fontsize=14)  # Make title bold and larger
plt.xlabel('Features', fontweight='bold', fontsize=14)  # Increase font size and make xlabel bold
plt.ylabel('Importance (%)', fontweight='bold', fontsize=14)  # Increase font size and make ylabel bold
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
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')  # Increase font size and make xlabel bold
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')  # Increase font size and make ylabel bold
    plt.title('ROC Curve', fontsize=14, fontweight='bold')  # Make title bold and larger
    plt.legend()
    plt.show()

# Plotting classification report
def plot_classification_report(classification_rep):
    # Splitting the classification report into lines
    lines = classification_rep.split('\n')

    # Extracting class names and metrics
    classes = []
    plotMat = []
    for line in lines[2:]:  # Excluding the first two lines (header)
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
    plt.title('Classification Report', fontsize=14, fontweight='bold')  # Make title bold and larger
    plt.colorbar()
    x_tick_marks = np.arange(len(classes))
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, classes, rotation=45, fontsize=12, fontweight='bold')  # Increase font size, make labels bold and rotate them
    plt.yticks(y_tick_marks, classes, fontsize=12, fontweight='bold')  # Increase font size and make labels bold
    plt.ylabel('Classes', fontsize=12, fontweight='bold')  # Increase font size and make ylabel bold
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')  # Increase font size and make xlabel bold
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
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')  # Make title bold and larger
plt.xlabel('Predicted', fontsize=12, fontweight='bold')  # Increase font size and make xlabel bold
plt.ylabel('True', fontsize=12, fontweight='bold')  # Increase font size and make ylabel bold
plt.show()

# Plot ROC curve
plot_roc_curve(y_test, y_pred_prob)

# Plot Classification Report
plot_classification_report(classification_rep)

# Given the healthiest values which its heart_disease_prob is zero
x1 = 29
x2 = 94
x3 = 100
x4 = 202
x5 = 0
x6 = 0
heart_disease_prob = 0  # Heart disease probability in percentage

# Given weights
w1 = 11.415535658641113
w2 = 11.134775996055545
w3 = 10.81141824129214
w4 = 18.34211551991339
w5 = 15.441381873669153
w6 = 13.701911603791928

# Calculate the sum of weights
sum_weights = w1 + w2 + w3 + w4 + w5 + w6

# Calculate b using the corrected formula with the sum of weights in the denominator
b = heart_disease_prob - ((w1 * (x1 - 29) / (77 - 29) + w2 * (x2 - 94) / (200 - 94) + w3 * (x3 - 100) / (564 - 100) - w4 * x4 / 6.2 + w5 * x5 / 3 + w6 * x6 / 3) / sum_weights)

print("The intercept term b is:", b)


# Given the wrost values of xi
x1_max = 77
x2_max = 200
x3_max = 564
x4_min = 71
x5_max = 6.2
x6_max = 3

# Define the values of w1, w2, w3, w4, w5, and w6
w1 = 11.415535658641113
w2 = 11.134775996055545
w3 = 10.81141824129214
w4 = 18.34211551991339
w5 = 15.441381873669153
w6 = 13.701911603791928

# Define the sum of weights
sum_weights = w1 + w2 + w3 + w4 + w5 + w6

# Define the intercept term b
b = 7.39

# Calculate the heart_disease_prob using the provided formula
heart_disease_prob = ((w1 * (x1_max - 29) / (77 - 29) + w2 * (x2_max - 94) / (200 - 94) + 
                      w3 * (x3_max - 100) / (564 - 100) - w4 * x4_min / 6.2 + 
                      w5 * x5_max / 3 + w6 * x6_max / 3) / sum_weights) + b

print("Heart Disease Probability:", heart_disease_prob)

# because for the wrost case the heart disease prob should be equal to one so we have to normalize above equation
heart_disease_prob = (((11.42 * (x1 - 29) / (77 - 29) + 11.13 * (x2 - 94) / (200 - 94) + 10.81 * (x3 - 100) / (564 - 100) - 18.34 * x4 / 6.2 + 15.44 * x5 / 3 + 13.70 * x6 / 3) / sum_weights) + b)
print("Normalized Heart Disease Probability:", heart_disease_prob)

from sympy import symbols, simplify

# Define symbolic variables
x1, x2, x3, x4, x5, x6 = symbols('x1 x2 x3 x4 x5 x6')

# Define the original expression
heart_disease_prob = (((11.42 * (x1 - 29) / (77 - 29) + 11.13 * (x2 - 94) / (200 - 94) + 
                      10.81 * (x3 - 100) / (564 - 100) - 18.34 * x4 / 6.2 + 
                      15.44 * x5 / 3 + 13.70 * x6 / 3) / 80.84) + 7.39)/5.77

# Simplify the expression
simplified_prob = simplify(heart_disease_prob)

print(simplified_prob)


# Simplified equation:

heart_disease_prob = 0.000510*x1 + 0.000225*x2 + 4.99e-5*x3 - 0.00634*x4 + 0.0110*x5 + 0.00979*x6 + 1.24

# Now by using the above equation we want to find the system dynamic as a discrete system
# between x1 to x6 we can modify and improve 3 of them by medication but we can not make changes in the others.
# So we can use [delat x2; delta x3; delta x5] as a input of the system.
# So we have to consider X = [x1; x2; x3; x4; x5; x6] and u = [delat x2; delta x3; delta x5]
# X(k+1) = A X(k) + B u(k) and y(k) = C X(k) + D u(k)
# A = I(6*6) B = I(3*3) C = [0.000510, 0.000225, 4.99E-5, -0.00634, 0.0110, 0.00979]

# now lets draw the rsponse of this dynamic system without controller. we will use u(0) = [-1; -1; -0.01] and X(0) =  [55; 130; 245; 150; 1.5; 1]
# X(0) is the current situation of the patient so it will be different for each patient

import numpy as np
import matplotlib.pyplot as plt

# Define system matrices and initial conditions
A = np.eye(6)
B = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]])
C = np.array([[0.000510, 0.000225, 4.99e-5, -0.00634, 0.0110, 0.00979]])
X0 = np.array([[77], [200], [564], [71], [6.2], [3]])
u0 = np.array([[-1], [-1], [-0.01]])

# Simulate the system
num_steps = 90
X = np.zeros((6, num_steps))
y = np.zeros((1, num_steps))

X[:, 0] = X0.flatten()
y[:, 0] = np.dot(C, X[:, 0]) + 1.24

for k in range(1, num_steps):
    X[:, k] = np.dot(A, X[:, k-1]) + np.dot(B, u0).flatten()
    y[:, k] = np.dot(C, X[:, k]) + 1.24

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(y.flatten(), label='Output y(k)')
plt.xlabel('Time Steps (k)', fontsize=12, fontweight='bold')  # Increase font size and make xlabel bold
plt.ylabel('Output y', fontsize=12, fontweight='bold')  # Increase font size and make ylabel bold
plt.title('System Response', fontsize=14, fontweight='bold')  # Make title bold and larger
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define system matrices and initial conditions
A = np.eye(6)
B = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]])
C = np.array([[0.000510, 0.000225, 4.99e-5, -0.00634, 0.0110, 0.00979]])
X0 = np.array([[77], [200], [564], [71], [6.2], [3]])
u0 = np.array([[-1], [-1], [-0.01]])


import numpy as np
import matplotlib.pyplot as plt

# Define system matrices and initial conditions
A = np.eye(6)
B = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]])
C = np.array([[0.000510, 0.000225, 4.99e-5, -0.00634, 0.0110, 0.00979]])
X0 = np.array([[77], [200], [564], [71], [6.2], [3]])
u0 = np.array([[-1], [-1], [-0.01]])

# Define desired reference value
r = 0  # Desired reference value

# Define proportional controller parameters
num_steps = 90
Kp_values = np.linspace(0.1, 1, 5)  # Values of Kp from 0 to 1

plt.figure(figsize=(10, 6))

for Kp in Kp_values:
    # Initialize arrays for simulation
    X = np.zeros((6, num_steps))
    y = np.zeros((1, num_steps))
    u = np.zeros((3, num_steps))

    # Initial conditions
    X[:, 0] = X0.flatten()
    y[:, 0] = np.dot(C, X[:, 0]) + 1.24

    # Simulate the system with proportional control
    for k in range(1, num_steps):
        error = r - y[:, k-1]
        u[:, k-1] = Kp * error
        X[:, k] = np.dot(A, X[:, k-1]) + np.dot(B, u[:, k-1])
        y[:, k] = np.dot(C, X[:, k]) + 1.24

    # Plot the response with proportional control for the current Kp value
    plt.plot(y.flatten(), label=f'Kp = {Kp:.2f}')

plt.xlabel('Time Steps (k)', fontsize=12, fontweight='bold')  # Increase font size and make xlabel bold
plt.ylabel('Output y', fontsize=12, fontweight='bold')  # Increase font size and make ylabel bold
plt.title('System Response with Proportional Control (Different Kp Values)', fontsize=14, fontweight='bold')  # Make title bold and larger
plt.legend()
plt.grid(True)
plt.show()
