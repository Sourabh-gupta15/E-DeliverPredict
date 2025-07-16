# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load data
df = pd.read_csv("E_Commerce.csv")
df.drop('ID', axis=1, inplace=True)

# Label Encoding
le = LabelEncoder()
for col in ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('Reached.on.Time_Y.N', axis=1)
y = df['Reached.on.Time_Y.N']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Model 1: Random Forest ------------------
rfc = RandomForestClassifier(criterion='gini', max_depth=8, min_samples_leaf=8,
                             min_samples_split=2, random_state=42)
rfc.fit(X_train, y_train)
joblib.dump(rfc, 'random_forest_model.pkl')

# ------------------ Model 2: Decision Tree ------------------
dtc = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=6,
                             min_samples_split=2, random_state=0)
dtc.fit(X_train, y_train)
joblib.dump(dtc, 'decision_tree_model.pkl')

# ------------------ Model 3: Logistic Regression ------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, 'logistic_regression_model.pkl')

# ------------------ Model 4: KNN ------------------
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, 'knn_model.pkl')

# Optional: Compare Accuracy
print("Random Forest:", accuracy_score(y_test, rfc.predict(X_test)))
print("Decision Tree:", accuracy_score(y_test, dtc.predict(X_test)))
print("Logistic Regression:", accuracy_score(y_test, lr.predict(X_test)))
print("KNN:", accuracy_score(y_test, knn.predict(X_test)))

print("\nModels saved successfully as .pkl files.")
