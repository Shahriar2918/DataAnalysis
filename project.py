import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Create Dataset
data = {
    'Age': [25, 30, np.nan, 45, 22, 35, 29, np.nan, 40, 50, 33, 26, 27, 38, 32, 31, 41, np.nan, 36, 48, 52, 28, 34, 39, np.nan, 37, 24, 49, 43, 21, 23, 46, 47, 44, 20, 42, 19, 18, 55, 53, 54, np.nan, 51, 60, 58, 57, 59, 56, 61, 62],
    'Salary': [50000, 60000, 55000, 65000, np.nan, 62000, 58000, 57000, 68000, 70000, 61000, 54000, 53000, 66000, 59000, 64000, np.nan, 56000, 63000, 69000, 72000, 52000, 60000, 67500, 51000, 62500, 51500, 70500, 68500, np.nan, 49500, 69500, 71000, 66000, 50500, 72000, 48000, 47500, 73500, 74000, 75000, 76000, np.nan, 77000, 78000, 79000, 80000, 81000, 82000, 83000],
    'Experience': [1, 3, 2, 5, 0, 4, 3, 2, 6, 8, 4, 1, 2, 5, 3, 4, 6, 2, 5, 7, 9, 1, 3, 6, 0, 4, 1, 8, 7, 0, 1, 7, 8, 6, 0, 7, 0, 0, 10, 9, 10, 11, 8, 12, 11, 10, 13, 12, 14, 15],
    'Department': ['HR', 'Finance', 'IT', 'Admin', 'HR', 'Finance', np.nan, 'Admin', 'IT', 'Finance', 'Admin', 'HR', 'Finance', 'IT', 'Admin', 'Finance', 'HR', 'Admin', 'IT', np.nan, 'HR', 'Finance', 'Admin', 'IT', 'Finance', 'Admin', 'HR', 'IT', 'Finance', 'Admin', 'HR', 'Finance', 'IT', 'Admin', 'HR', 'Finance', 'Admin', 'IT', 'Finance', 'HR', 'Admin', 'IT', 'Finance', 'Admin', 'HR', 'Finance', 'IT', 'Admin', 'HR', 'Finance'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', np.nan, 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', np.nan, 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', np.nan, 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Status': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Handle Missing Values
df.fillna({
    'Age': df['Age'].mean(),
    'Salary': df['Salary'].mean(),
    'Department': df['Department'].mode()[0],
    'Gender': df['Gender'].mode()[0]
}, inplace=True)

# Step 3: Convert Categorical Data
df = pd.get_dummies(df, columns=['Department', 'Gender'], drop_first=True)

# Step 4: Statistical Analysis
print("Mean:\n", df.mean())
print("\nMedian:\n", df.median())
print("\nMode:\n", df.mode().iloc[0])

# Step 5: Data Visualization
sns.pairplot(df)
plt.savefig('data_visualization.png')
plt.show()

# Step 6: Machine Learning
X = df.drop('Status', axis=1)
y = df['Status']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, rf_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
