import pandas as pd

# Load datasets
symptoms = pd.read_csv("symptoms_disease.csv")
medicine = pd.read_csv("medicine_data.csv")
hospital = pd.read_csv("hospital_data.csv")

from sklearn.preprocessing import LabelEncoder

# Label Encoders
le_fever = LabelEncoder()
le_cough = LabelEncoder()
le_fatigue = LabelEncoder()
le_breath = LabelEncoder()
le_gender = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_outcome = LabelEncoder()
le_disease = LabelEncoder()

# Encoding
symptoms['Fever'] = le_fever.fit_transform(symptoms['Fever'])
symptoms['Cough'] = le_cough.fit_transform(symptoms['Cough'])
symptoms['Fatigue'] = le_fatigue.fit_transform(symptoms['Fatigue'])
symptoms['Difficulty Breathing'] = le_breath.fit_transform(symptoms['Difficulty Breathing'])
symptoms['Gender'] = le_gender.fit_transform(symptoms['Gender'])
symptoms['Blood Pressure'] = le_bp.fit_transform(symptoms['Blood Pressure'])
symptoms['Cholesterol Level'] = le_chol.fit_transform(symptoms['Cholesterol Level'])
symptoms['Outcome Variable'] = le_outcome.fit_transform(symptoms['Outcome Variable'])
symptoms['Disease'] = le_disease.fit_transform(symptoms['Disease'])

from sklearn.model_selection import train_test_split

# Features and Target
X = symptoms.drop('Disease', axis=1)
y = symptoms['Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model training completed")

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Medicine keywords
disease_medicine = {
    "Asthma": ["Montelukast", "Salbutamol", "Budesonide"],
    "Common Cold": ["Paracetamol", "Cetirizine", "Dextromethorphan"],
    "Influenza": ["Oseltamivir", "Paracetamol", "Ibuprofen"],
    "Pneumonia": ["Azithromycin", "Amoxicillin", "Levofloxacin"]
}

print("\n==============================")
print(" AI HEALTH ASSISTANT SYSTEM ")
print("==============================")

while True:

    print("\n1 Predict Disease")
    print("2 Search Medicines")
    print("3 Find Hospitals")
    print("4 Exit")

    choice = input("\nEnter your choice: ").strip()

    # OPTION 1
    if choice == "1":

        print("\n--- Enter Patient Symptoms ---")

        fever = input("Fever (Yes/No): ").strip().capitalize()
        cough = input("Cough (Yes/No): ").strip().capitalize()
        fatigue = input("Fatigue (Yes/No): ").strip().capitalize()
        breathing = input("Difficulty Breathing (Yes/No): ").strip().capitalize()
        age = int(input("Age: "))
        gender = input("Gender (Male/Female): ").strip().capitalize()
        bp = input("Blood Pressure (Low/Normal/High): ").strip().capitalize()
        chol = input("Cholesterol Level (Normal/High): ").strip().capitalize()
        outcome = input("Outcome Variable (Positive/Negative): ").strip().capitalize()

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
        predicted_disease = le_disease.inverse_transform(prediction)

        print("\nPredicted Disease:", predicted_disease[0])

    # OPTION 2
    elif choice == "2":

        disease_name = input("\nEnter Disease Name: ").strip().capitalize()

        print("\nRecommended Medicines:")

        if disease_name in disease_medicine:

            keywords = disease_medicine[disease_name]

            filtered = medicine[
                medicine["short_composition1"].str.contains('|'.join(keywords), case=False, na=False)
            ]

            print(filtered[["name", "price(₹)", "short_composition1"]].head(5))

        else:
            print("No medicines found for this disease")

    # OPTION 3
    elif choice == "3":

        city = input("\nEnter your City: ").strip().capitalize()

        filtered_hospitals = hospital[
            hospital["City"].str.contains(city, case=False, na=False)
        ]

        print("\nTop Hospitals in", city)

        if len(filtered_hospitals) > 0:
            print(filtered_hospitals[["Hospital", "City", "LocalAddress", "Pincode"]].head(5))
        else:
            print("No hospitals found")

    # OPTION 4
    elif choice == "4":
        print("\nThank you for using AI Health Assistant")
        break

    else:
        print("\nInvalid Choice. Try again.")
        