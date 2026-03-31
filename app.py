import streamlit as st
import pandas as pd
import requests
from streamlit_lottie import st_lottie

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------- LOTTIE FUNCTION ----------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---------------- UI SETTINGS ----------------
st.set_page_config(page_title="AI Health Assistant", layout="wide")

st.title("🤖 AI Health Assistant")
lottie = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json")
st_lottie(lottie, height=300)

st.markdown("Get instant disease prediction, medicines & hospitals")

# ---------------- LOAD DATA ----------------
symptoms = pd.read_csv("symptoms_disease.csv")
medicine = pd.read_csv("medicine_data.csv")
hospital = pd.read_csv("hospital_data.csv")

# ---------------- ENCODING ----------------
le_fever = LabelEncoder()
le_cough = LabelEncoder()
le_fatigue = LabelEncoder()
le_breath = LabelEncoder()
le_gender = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_outcome = LabelEncoder()
le_disease = LabelEncoder()

symptoms['Fever'] = le_fever.fit_transform(symptoms['Fever'])
symptoms['Cough'] = le_cough.fit_transform(symptoms['Cough'])
symptoms['Fatigue'] = le_fatigue.fit_transform(symptoms['Fatigue'])
symptoms['Difficulty Breathing'] = le_breath.fit_transform(symptoms['Difficulty Breathing'])
symptoms['Gender'] = le_gender.fit_transform(symptoms['Gender'])
symptoms['Blood Pressure'] = le_bp.fit_transform(symptoms['Blood Pressure'])
symptoms['Cholesterol Level'] = le_chol.fit_transform(symptoms['Cholesterol Level'])
symptoms['Outcome Variable'] = le_outcome.fit_transform(symptoms['Outcome Variable'])
symptoms['Disease'] = le_disease.fit_transform(symptoms['Disease'])

# ---------------- MODEL ----------------
X = symptoms.drop('Disease', axis=1)
y = symptoms['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Predict Disease", "Medicines", "Hospitals"])

# ---------------- OPTION 1 ----------------
if option == "Predict Disease":

    st.header("🧠 Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        fever = st.selectbox("Fever", ["Yes", "No"])
        cough = st.selectbox("Cough", ["Yes", "No"])
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])

    with col2:
        age = st.number_input("Age", 1, 100)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bp = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
        chol = st.selectbox("Cholesterol Level", ["Normal", "High"])
        outcome = st.selectbox("Outcome", ["Positive", "Negative"])

    if st.button("🔍 Predict"):

        input_data = [
            le_fever.transform([fever])[0],
            le_cough.transform([cough])[0],
            le_fatigue.transform([fatigue])[0],
            le_breath.transform([breathing])[0],
            age,
            le_gender.transform([gender])[0],
            le_bp.transform([bp])[0],
            le_chol.transform([chol])[0],
            le_outcome.transform([outcome])[0]
        ]

        prediction = model.predict([input_data])
        disease = le_disease.inverse_transform(prediction)[0]

        # ✅ Stylish Output
        st.markdown(f"## 🧠 Predicted Disease: **{disease}**")
        st.balloons()

        # ✅ Medicines Auto Show
        st.subheader("💊 Recommended Medicines")

        filtered = medicine[
            medicine["name"].str.contains(disease, case=False, na=False)
        ]

        if len(filtered) > 0:
            st.dataframe(filtered[["name", "price(₹)", "short_composition1"]].head(5))
        else:
            st.write("General medicines:")
            st.dataframe(medicine.head(5))

        # ✅ Warning
        st.info("⚠️ This is AI-based prediction. Please consult a doctor.")

# ---------------- OPTION 2 ----------------
elif option == "Medicines":

    st.header("💊 Medicine Recommendation")

    disease_input = st.text_input("Enter Disease Name")

    if st.button("Search Medicines"):

        filtered = medicine[
            medicine["name"].str.contains(disease_input, case=False, na=False)
        ]

        if len(filtered) > 0:
            st.dataframe(filtered[["name", "price(₹)", "short_composition1"]].head(5))
        else:
            st.warning("No medicines found")

# ---------------- OPTION 3 ----------------
elif option == "Hospitals":

    st.header("🏥 Find Hospitals")

    city = st.text_input("Enter City")

    if st.button("Search Hospitals"):

        filtered = hospital[
            hospital["City"].str.contains(city, case=False, na=False)
        ]

        if len(filtered) > 0:
            st.dataframe(filtered[["Hospital", "City", "LocalAddress", "Pincode"]].head(5))
        else:
            st.warning("No hospitals found")


