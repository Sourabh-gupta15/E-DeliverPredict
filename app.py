import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="E-Commerce Delivery Prediction", layout="wide")
st.title("📦 E-Commerce Product Delivery Prediction Dashboard")

# Load dataset (same one used for training, minus ID column)
df = pd.read_csv("E_Commerce.csv")
df.drop("ID", axis=1, inplace=True)

# Label encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop("Reached.on.Time_Y.N", axis=1)
y = df["Reached.on.Time_Y.N"]

# Sidebar model selection
st.sidebar.header("🔍 Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ("Random Forest", "Decision Tree", "Logistic Regression", "KNN")
)

# Load selected model
model_map = {
    "Random Forest": "random_forest_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "KNN": "knn_model.pkl"
}

model = joblib.load(model_map[model_choice])
y_pred = model.predict(X)

# Display performance
st.subheader(f"📈 Model Evaluation: {model_choice}")
acc = accuracy_score(y, y_pred)
st.metric("Accuracy", f"{acc:.2%}")

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().round(2))

# Confusion Matrix
st.subheader("🔄 Confusion Matrix")
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["On Time", "Late"], yticklabels=["On Time", "Late"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)
