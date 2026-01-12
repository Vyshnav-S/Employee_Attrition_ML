import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ===============================
# CONFIG
# ===============================
DATA_PATH = "employee_attrition.csv"
MODEL_PATH = "attrition_pipeline.pkl"

# ===============================
# TRAIN MODEL (ONLY IF NEEDED)
# ===============================
@st.cache_resource
def train_and_save_model():

    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    num_features = [
        "Age", "MonthlyIncome", "YearsAtCompany",
        "WorkLifeBalance", "JobSatisfaction", "PerformanceRating"
    ]

    cat_features = [
        "Gender", "Department", "JobRole", "OverTime"
    ]

    text_feature = "Feedback"

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=100))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features),
        ("text", text_pipeline, text_feature)
    ])

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("pca", PCA(n_components=20)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nMODEL PERFORMANCE\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

# ===============================
# LOAD MODEL
# ===============================
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_and_save_model()

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Employee Attrition Predictor")

st.title("👩‍💼 Employee Attrition Prediction")

st.subheader("Enter Employee Information")

age = st.number_input("Age", 18, 65)
income = st.number_input("Monthly Income", 0)
years = st.number_input("Years At Company", 0)
wlb = st.selectbox("Work Life Balance", [1, 2, 3, 4])
js = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
pr = st.selectbox("Performance Rating", [1, 2, 3, 4])

gender = st.selectbox("Gender", ["Male", "Female"])
dept = st.selectbox("Department", ["IT", "Sales", "HR", "Operations"])
role = st.selectbox("Job Role", ["Engineer", "Analyst", "Technician"])
ot = st.selectbox("OverTime", ["Yes", "No"])

feedback = st.text_area("Employee Feedback")

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "WorkLifeBalance": wlb,
        "JobSatisfaction": js,
        "PerformanceRating": pr,
        "Gender": gender,
        "Department": dept,
        "JobRole": role,
        "OverTime": ot,
        "Feedback": feedback
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠ High Attrition Risk ({prob:.2%})")
    else:
        st.success(f"✅ Low Attrition Risk ({prob:.2%})")
