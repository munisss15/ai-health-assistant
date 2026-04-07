import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Health Assistant", layout="wide")

st.title("🤖 AI Health Assistant Pro")
st.markdown("### Predict Disease • Find Medicines • Locate Hospitals")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    symptoms = pd.read_csv("symptoms_disease.csv")
    medicine = pd.read_csv("medicine_data.csv")
    hospital = pd.read_csv("hospital_data.csv")
    return symptoms, medicine, hospital

symptoms, medicine, hospital = load_data()

# ---------------- ENCODING ----------------
le = {}
columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
           'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']

for col in columns:
    le[col] = LabelEncoder()
    symptoms[col] = le[col].fit_transform(symptoms[col])

le_disease = LabelEncoder()
symptoms['Disease'] = le_disease.fit_transform(symptoms['Disease'])

# ---------------- MODEL ----------------
X = symptoms.drop('Disease', axis=1)
y = symptoms['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🧠 Prediction", "💊 Medicines", "🏥 Hospitals"])

# ================== TAB 1 ==================
with tab1:
    st.subheader("Enter Symptoms")

    col1, col2 = st.columns(2)

    with col1:
        fever = st.selectbox("Fever", ["Yes", "No"])
        cough = st.selectbox("Cough", ["Yes", "No"])
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])

    with col2:
        age = st.slider("Age", 1, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bp = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
        chol = st.selectbox("Cholesterol", ["Normal", "High"])
        outcome = st.selectbox("Condition", ["Positive", "Negative"])

    if st.button("🔍 Predict Disease"):
        try:
            input_data = [
                le['Fever'].transform([fever])[0],
                le['Cough'].transform([cough])[0],
                le['Fatigue'].transform([fatigue])[0],
                le['Difficulty Breathing'].transform([breathing])[0],
                age,
                le['Gender'].transform([gender])[0],
                le['Blood Pressure'].transform([bp])[0],
                le['Cholesterol Level'].transform([chol])[0],
                le['Outcome Variable'].transform([outcome])[0]
            ]

            prediction = model.predict([input_data])
            disease = le_disease.inverse_transform(prediction)[0]

            st.success(f"🧠 Predicted Disease: {disease}")
            st.balloons()

            # Medicines auto show
            st.subheader("💊 Recommended Medicines")

            filtered = medicine[
                medicine["name"].str.contains(disease, case=False, na=False)
            ]

            if len(filtered) > 0:
                st.dataframe(filtered[["name", "price(₹)", "short_composition1"]].head(5))
            else:
                st.warning("No exact match found. Showing general medicines.")
                st.dataframe(medicine.head(5))

        except Exception as e:
            st.error("Error in prediction. Check your data.")

# ================== TAB 2 ==================
with tab2:
    st.subheader("Search Medicines")

    disease_input = st.text_input("Enter Disease")

    if st.button("Search"):
        filtered = medicine[
            medicine["name"].str.contains(disease_input, case=False, na=False)
        ]

        if len(filtered) > 0:
            st.dataframe(filtered[["name", "price(₹)", "short_composition1"]])
        else:
            st.warning("No medicines found")

# ================== TAB 3 ==================
with tab3:
    st.subheader("Find Hospitals")

    city = st.text_input("Enter City Name")

    if st.button("Find"):
        filtered = hospital[
            hospital["City"].str.contains(city, case=False, na=False)
        ]

        if len(filtered) > 0:
            st.dataframe(filtered[["Hospital", "City", "LocalAddress", "Pincode"]])
        else:
            st.warning("No hospitals found")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ This is AI-based prediction. Consult a doctor.")


