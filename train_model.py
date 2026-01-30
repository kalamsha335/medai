import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------
# EXPERT DATASET GENERATOR
# --------------------------
# This creates many symptom combinations per disease (synthetic but realistic).
# You can later replace with real clinical dataset if you have one.

disease_bank = {
    "viral fever": [
        "fever", "high fever", "body pain", "headache", "weakness", "fatigue", "chills"
    ],
    "common cold": [
        "cough", "sore throat", "runny nose", "sneezing", "nasal congestion", "mild fever"
    ],
    "influenza": [
        "high fever", "body pain", "chills", "dry cough", "fatigue", "headache"
    ],
    "malaria": [
        "high fever", "chills", "sweating", "body pain", "vomiting", "weakness"
    ],
    "dengue": [
        "high fever", "severe headache", "joint pain", "rash", "body pain", "fatigue"
    ],
    "typhoid": [
        "fever", "weakness", "stomach pain", "loss of appetite", "headache"
    ],
    "sinusitis": [
        "headache", "facial pain", "nasal congestion", "runny nose", "post nasal drip"
    ],
    "migraine": [
        "headache", "nausea", "vomiting", "sensitivity to light", "blurred vision"
    ],
    "tension headache": [
        "headache", "stress", "neck pain", "fatigue"
    ],
    "asthma": [
        "breathlessness", "wheezing", "cough", "chest tightness"
    ],
    "bronchitis": [
        "cough", "chest pain", "mucus", "fatigue", "mild fever"
    ],
    "pneumonia": [
        "high fever", "cough", "breathlessness", "chest pain", "fatigue"
    ],
    "respiratory infection": [
        "cough", "breathlessness", "chest pain", "fever"
    ],
    "heart problem": [
        "chest pain", "breathlessness", "sweating", "left arm pain", "dizziness"
    ],
    "hypertension": [
        "headache", "dizziness", "blurred vision", "chest discomfort"
    ],
    "diabetes": [
        "frequent urination", "thirst", "fatigue", "weight loss", "slow healing"
    ],
    "hypoglycemia": [
        "sweating", "shaking", "dizziness", "hunger", "confusion"
    ],
    "anemia": [
        "fatigue", "weakness", "pale skin", "shortness of breath", "dizziness"
    ],
    "dehydration": [
        "dry mouth", "thirst", "dizziness", "dark urine", "fatigue"
    ],
    "food poisoning": [
        "vomiting", "diarrhea", "stomach cramps", "nausea", "fever"
    ],
    "gastritis": [
        "stomach pain", "nausea", "vomiting", "acidity", "burning stomach"
    ],
    "acid reflux": [
        "acidity", "heartburn", "burning chest", "burping", "sour taste"
    ],
    "constipation": [
        "hard stool", "stomach pain", "bloating", "no bowel movement"
    ],
    "diarrhea": [
        "loose motion", "watery stool", "stomach cramps", "dehydration"
    ],
    "uti": [
        "burning urination", "frequent urination", "lower abdominal pain", "fever"
    ],
    "kidney stone": [
        "severe back pain", "painful urination", "blood in urine", "nausea"
    ],
    "hepatitis": [
        "yellow skin", "dark urine", "fatigue", "nausea", "loss of appetite"
    ],
    "appendicitis": [
        "lower right abdominal pain", "vomiting", "fever", "loss of appetite"
    ],
    "allergy": [
        "itching", "rash", "sneezing", "watery eyes", "swelling"
    ],
    "skin infection": [
        "redness", "swelling", "pain", "pus", "fever"
    ],
    "eczema": [
        "itching", "dry skin", "rash", "skin redness"
    ],
    "acne": [
        "pimples", "oily skin", "face bumps"
    ],
    "conjunctivitis": [
        "red eye", "itchy eye", "watery eye", "eye discharge"
    ],
    "ear infection": [
        "ear pain", "fever", "hearing loss", "ear discharge"
    ],
    "tonsillitis": [
        "sore throat", "fever", "difficulty swallowing", "swollen tonsils"
    ],
    "dental infection": [
        "tooth pain", "swelling", "bad breath", "fever"
    ],
    "arthritis": [
        "joint pain", "swelling", "stiffness", "reduced movement"
    ],
    "spine issue": [
        "back pain", "stiffness", "posture pain", "neck pain"
    ],
    "muscle strain": [
        "muscle pain", "swelling", "tenderness", "movement pain"
    ],
    "fracture": [
        "severe pain", "swelling", "unable to move", "deformity"
    ],
    "anxiety": [
        "panic", "fast heartbeat", "sweating", "fear", "breathlessness"
    ],
    "depression": [
        "sadness", "loss of interest", "fatigue", "sleep issues"
    ],
    "insomnia": [
        "sleep problem", "difficulty sleeping", "tiredness", "stress"
    ],
    "heat stroke": [
        "high fever", "hot skin", "confusion", "dizziness", "vomiting"
    ],
    "cold allergy": [
        "sneezing", "runny nose", "watery eyes", "itching"
    ],
    "covid-19": [
        "fever", "dry cough", "loss of smell", "breathlessness", "fatigue"
    ],
    "chickenpox": [
        "fever", "itchy rash", "blisters", "body pain"
    ],
    "measles": [
        "fever", "rash", "cough", "runny nose", "red eyes"
    ],
    "gout": [
        "joint pain", "swelling", "redness", "big toe pain"
    ],
    "thyroid disorder": [
        "fatigue", "weight gain", "hair loss", "cold intolerance"
    ],
    "pcos": [
        "irregular periods", "weight gain", "acne", "hair growth"
    ],
    "pregnancy nausea": [
        "nausea", "vomiting", "missed period", "fatigue"
    ],
}

def generate_rows(disease, symptoms):
    rows = []
    # Create multiple combinations
    base = symptoms[:]
    # 1 symptom
    for s in base[:4]:
        rows.append((s, disease))
    # 2 symptoms
    for i in range(min(5, len(base))):
        for j in range(i+1, min(6, len(base))):
            rows.append((f"{base[i]} {base[j]}", disease))
    # 3 symptoms
    for i in range(min(4, len(base))):
        rows.append((f"{base[i]} {base[(i+1)%len(base)]} {base[(i+2)%len(base)]}", disease))
    # Full phrase
    rows.append((" ".join(base[:6]), disease))
    return rows

all_rows = []
for dis, sym_list in disease_bank.items():
    all_rows.extend(generate_rows(dis, sym_list))

df = pd.DataFrame(all_rows, columns=["symptoms", "disease"])
df.drop_duplicates(inplace=True)

# Save dataset (optional)
df.to_csv("dataset.csv", index=False)
print(f"✅ Generated dataset rows: {len(df)}")
print("✅ Saved dataset.csv")

# Train model
X = df["symptoms"].astype(str)
y = df["disease"].astype(str)

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=2000)
model.fit(X_vec, y)

os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🎉 SUCCESS: model/model.pkl and model/vectorizer.pkl created")
