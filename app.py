from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
from datetime import datetime
import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"

# MongoDB
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["MediAI_DB"]
users_col = db["users"]
chats_col = db["chats"]
consult_col = db["consultations"]  # NEW

# Load AI model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------
# Doctor Mapping
# -------------------------
doctor_map = {
    "viral fever": "General Physician",
    "common cold": "ENT Specialist",
    "influenza": "General Physician",
    "malaria": "General Physician",
    "dengue": "General Physician",
    "typhoid": "General Physician",
    "sinusitis": "ENT Specialist",
    "migraine": "Neurologist",
    "tension headache": "General Physician",
    "asthma": "Pulmonologist",
    "bronchitis": "Pulmonologist",
    "pneumonia": "Pulmonologist",
    "respiratory infection": "Pulmonologist",
    "heart problem": "Cardiologist",
    "hypertension": "Cardiologist / General Physician",
    "diabetes": "Diabetologist",
    "hypoglycemia": "General Physician",
    "anemia": "General Physician",
    "dehydration": "General Physician",
    "food poisoning": "General Physician",
    "gastritis": "Gastroenterologist",
    "acid reflux": "Gastroenterologist",
    "constipation": "General Physician",
    "diarrhea": "General Physician",
    "uti": "Urologist",
    "kidney stone": "Urologist",
    "hepatitis": "Gastroenterologist",
    "appendicitis": "General Surgeon",
    "allergy": "Dermatologist / ENT",
    "skin infection": "Dermatologist",
    "eczema": "Dermatologist",
    "acne": "Dermatologist",
    "conjunctivitis": "Ophthalmologist",
    "ear infection": "ENT Specialist",
    "tonsillitis": "ENT Specialist",
    "dental infection": "Dentist",
    "arthritis": "Orthopedic / Rheumatologist",
    "spine issue": "Orthopedic",
    "muscle strain": "Orthopedic",
    "fracture": "Orthopedic (Emergency)",
    "anxiety": "Psychiatrist / Counselor",
    "depression": "Psychiatrist / Counselor",
    "insomnia": "General Physician",
    "heat stroke": "Emergency / Hospital",
    "cold allergy": "ENT Specialist",
    "covid-19": "General Physician",
    "chickenpox": "General Physician",
    "measles": "General Physician",
    "gout": "Rheumatologist",
    "thyroid disorder": "Endocrinologist",
    "pcos": "Gynecologist",
    "pregnancy nausea": "Gynecologist",
}

# -------------------------
# Text Normalization
# -------------------------
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", " ").replace("_", " ")

    mapping = {
        "heart pain": "chest pain",
        "heartpain": "chest pain",
        "chestpain": "chest pain",
        "breathing trouble": "breathlessness",
        "breathless": "breathlessness",
        "loose motion": "diarrhea",
        "stomach ache": "stomach pain",
        "throwing up": "vomiting",
        "bp high": "hypertension",
        "sugar high": "diabetes",
        "teeth pain": "tooth pain",
        "toothache": "tooth pain",
    }

    for k, v in mapping.items():
        text = text.replace(k, v)

    return text

# -------------------------
# Emergency Check
# -------------------------
def emergency_check(text: str) -> bool:
    text = text.lower()
    emergency_signs = [
        "chest pain", "breathlessness", "unconscious", "severe bleeding",
        "stroke", "seizure", "paralysis", "not responding"
    ]
    return any(word in text for word in emergency_signs)

# -------------------------
# Precautions (Virtual Doctor)
# -------------------------
def get_precaution(disease_name: str) -> str:
    disease_name = disease_name.lower().strip()

    fever_group = ["viral fever", "influenza", "malaria", "dengue", "typhoid", "covid-19", "measles", "chickenpox"]
    stomach_group = ["gastritis", "food poisoning", "acid reflux", "diarrhea", "constipation", "appendicitis"]
    respiratory_group = ["common cold", "bronchitis", "pneumonia", "respiratory infection", "asthma", "sinusitis", "tonsillitis"]
    skin_group = ["allergy", "eczema", "skin infection", "acne"]
    urinary_group = ["uti", "kidney stone"]
    bone_group = ["arthritis", "spine issue", "muscle strain", "fracture"]
    mental_group = ["anxiety", "depression", "insomnia"]

    if disease_name in ["heart problem", "stroke"]:
        return "🚨 Emergency: Go to hospital immediately if severe chest pain, sweating, breathlessness, weakness, or fainting occurs."

    if disease_name in fever_group:
        return "Rest, drink fluids, monitor temperature, and consult doctor if fever lasts more than 2 days."

    if disease_name in stomach_group:
        return "Drink ORS/water, eat light food, avoid spicy/oily foods, and consult doctor if vomiting/diarrhea continues."

    if disease_name in respiratory_group:
        return "Steam inhalation, warm fluids, avoid smoke/cold drinks, and consult doctor if breathing difficulty occurs."

    if disease_name in skin_group:
        return "Avoid allergens, keep skin clean, do not scratch, and consult dermatologist if swelling spreads."

    if disease_name in urinary_group:
        return "Drink more water, maintain hygiene, avoid holding urine, and consult doctor for antibiotics if burning persists."

    if disease_name in bone_group:
        return "Rest the affected area, avoid heavy work, use warm compress, and consult orthopedic if pain is severe."

    if disease_name in mental_group:
        return "Reduce stress, follow a proper sleep routine, and consult a professional if symptoms persist."

    return "Please consult a doctor for proper diagnosis. Avoid self-medication."

# -------------------------
# Triage (Severity)
# -------------------------
def triage_level(symptoms: str) -> dict:
    s = symptoms.lower()
    score = 0

    if "chest pain" in s:
        score += 60
    if "breathlessness" in s:
        score += 50
    if "unconscious" in s:
        score += 80
    if "seizure" in s:
        score += 80
    if "severe bleeding" in s:
        score += 90
    if "high fever" in s or "fever" in s:
        score += 20
    if "vomiting" in s:
        score += 15
    if "diarrhea" in s:
        score += 15

    if score >= 80:
        return {"level": "RED", "advice": "🚨 EMERGENCY: Visit hospital immediately."}
    elif score >= 40:
        return {"level": "YELLOW", "advice": "⚠️ URGENT: Consult doctor within 24 hours."}
    else:
        return {"level": "GREEN", "advice": "✅ Mild: Home care + monitor symptoms."}

# -------------------------
# Language Output (English + Tamil)
# -------------------------
def bilingual(text_en: str, text_ta: str) -> str:
    return f"{text_en}\n\nதமிழ்:\n{text_ta}"

# -------------------------
# Virtual Doctor Conversation State
# -------------------------
def init_consultation():
    return {
        "step": 1,
        "symptoms": "",
        "duration": "",
        "severity": "",
        "age": "",
        "gender": "",
        "created_at": datetime.now()
    }

def get_next_question(state):
    step = state.get("step", 1)

    if step == 1:
        return bilingual(
            "Hello! I am your virtual doctor. Please tell your main symptoms.",
            "வணக்கம்! நான் உங்கள் மெய்நிகர் மருத்துவர். உங்கள் முக்கிய அறிகுறிகளை சொல்லுங்கள்."
        )

    if step == 2:
        return bilingual(
            "How long have you had these symptoms? (hours/days)",
            "இந்த அறிகுறிகள் எத்தனை நேரம்/நாட்களாக உள்ளது?"
        )

    if step == 3:
        return bilingual(
            "How severe is the problem? (mild / moderate / severe)",
            "இந்த பிரச்சனை எவ்வளவு தீவிரம்? (குறைவு / மிதமான / அதிகம்)"
        )

    if step == 4:
        return bilingual(
            "Please enter your age.",
            "உங்கள் வயதை உள்ளிடுங்கள்."
        )

    if step == 5:
        return bilingual(
            "Gender? (male / female)",
            "பாலினம்? (ஆண் / பெண்)"
        )

    return None

def finalize_virtual_doctor(symptoms: str):
    # Emergency override
    if emergency_check(symptoms):
        triage = {"level": "RED", "advice": "🚨 EMERGENCY: Go to nearest hospital immediately."}
        reply_text = bilingual(
            "Emergency signs detected. Please go to hospital immediately.",
            "அவசர அறிகுறிகள் கண்டறியப்பட்டது. உடனே அருகிலுள்ள மருத்துவமனைக்கு செல்லுங்கள்."
        )
        return {
            "emergency": True,
            "triage": triage,
            "top3": [{"disease": "EMERGENCY", "confidence": 100.0}],
            "doctor": "Emergency / Hospital",
            "precaution": reply_text
        }

    X_vec = vectorizer.transform([symptoms])
    probs = model.predict_proba(X_vec)[0]
    classes = model.classes_

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"disease": classes[i].upper(), "confidence": round(float(probs[i] * 100), 2)} for i in top_idx]

    top_disease = classes[top_idx[0]]
    doctor = doctor_map.get(top_disease, "General Physician")
    precaution_en = get_precaution(top_disease)

    triage = triage_level(symptoms)

    precaution_text = bilingual(
        f"Doctor Advice: {precaution_en}\n\nTriage: {triage['level']} - {triage['advice']}",
        f"மருத்துவர் ஆலோசனை: {precaution_en}\n\nஅவசர நிலை: {triage['level']} - {triage['advice']}"
    )

    return {
        "emergency": False,
        "triage": triage,
        "top3": top3,
        "doctor": doctor,
        "precaution": precaution_text
    }

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        if users_col.find_one({"email": email}):
            return "Email already exists! Try login."

        users_col.insert_one({
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
            "created_at": datetime.now()
        })
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = users_col.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user"] = user["email"]
            session["name"] = user["name"]

            # start new consultation
            consult_col.delete_many({"email": session["user"]})
            consult_col.insert_one({"email": session["user"], "state": init_consultation()})

            return redirect(url_for("dashboard"))

        return "Invalid login details!"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", name=session["name"])

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    msg = normalize_text(data.get("message", ""))

    consult = consult_col.find_one({"email": session["user"]})
    if not consult:
        consult_col.insert_one({"email": session["user"], "state": init_consultation()})
        consult = consult_col.find_one({"email": session["user"]})

    state = consult["state"]
    step = state.get("step", 1)

    # Step-wise doctor conversation
    if step == 1:
        state["symptoms"] = msg
        state["step"] = 2
        consult_col.update_one({"email": session["user"]}, {"$set": {"state": state}})
        bot_msg = get_next_question(state)

        reply = {
            "emergency": False,
            "virtual_mode": True,
            "step": state["step"],
            "bot_question": bot_msg,
            "top3": [],
            "doctor": "",
            "precaution": ""
        }
        return jsonify(reply)

    if step == 2:
        state["duration"] = msg
        state["step"] = 3
        consult_col.update_one({"email": session["user"]}, {"$set": {"state": state}})
        bot_msg = get_next_question(state)

        reply = {
            "emergency": False,
            "virtual_mode": True,
            "step": state["step"],
            "bot_question": bot_msg,
            "top3": [],
            "doctor": "",
            "precaution": ""
        }
        return jsonify(reply)

    if step == 3:
        state["severity"] = msg
        state["step"] = 4
        consult_col.update_one({"email": session["user"]}, {"$set": {"state": state}})
        bot_msg = get_next_question(state)

        reply = {
            "emergency": False,
            "virtual_mode": True,
            "step": state["step"],
            "bot_question": bot_msg,
            "top3": [],
            "doctor": "",
            "precaution": ""
        }
        return jsonify(reply)

    if step == 4:
        state["age"] = msg
        state["step"] = 5
        consult_col.update_one({"email": session["user"]}, {"$set": {"state": state}})
        bot_msg = get_next_question(state)

        reply = {
            "emergency": False,
            "virtual_mode": True,
            "step": state["step"],
            "bot_question": bot_msg,
            "top3": [],
            "doctor": "",
            "precaution": ""
        }
        return jsonify(reply)

    if step == 5:
        state["gender"] = msg
        state["step"] = 6  # finalize
        consult_col.update_one({"email": session["user"]}, {"$set": {"state": state}})

        full_symptoms = f"{state['symptoms']} duration {state['duration']} severity {state['severity']} age {state['age']} gender {state['gender']}"
        final_reply = finalize_virtual_doctor(full_symptoms)

        chats_col.insert_one({
            "email": session["user"],
            "symptoms": full_symptoms,
            "reply": final_reply,
            "timestamp": datetime.now()
        })

        # restart consultation after final
        consult_col.update_one({"email": session["user"]}, {"$set": {"state": init_consultation()}})

        return jsonify(final_reply)

    # fallback
    consult_col.update_one({"email": session["user"]}, {"$set": {"state": init_consultation()}})
    return jsonify({"error": "Session reset. Please start again."})

@app.route("/history")
def history():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    records = chats_col.find({"email": session["user"]}).sort("timestamp", -1).limit(20)
    out = []
    for r in records:
        out.append({
            "symptoms": r["symptoms"],
            "reply": r["reply"],
            "time": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify(out)

@app.route("/download_report")
def download_report():
    if "user" not in session:
        return redirect(url_for("login"))

    last_chat = chats_col.find_one({"email": session["user"]}, sort=[("timestamp", -1)])
    if not last_chat:
        return "No history found!"

    filename = f"Medical_Report_{session['name']}.pdf"
    filepath = os.path.join("static", filename)

    c = canvas.Canvas(filepath, pagesize=letter)
    c.setFont("Helvetica", 12)

    c.drawString(50, 750, "MediAI - Virtual Doctor Report")
    c.drawString(50, 730, f"Patient Name: {session['name']}")
    c.drawString(50, 710, f"Email: {session['user']}")
    c.drawString(50, 690, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, 650, "Patient Inputs:")
    c.drawString(70, 630, last_chat["symptoms"][:90])

    c.drawString(50, 590, "Top Predictions:")
    y = 570
    for item in last_chat["reply"]["top3"]:
        c.drawString(70, y, f"- {item['disease']} ({item['confidence']}%)")
        y -= 20

    c.drawString(50, y - 10, "Recommended Doctor:")
    c.drawString(70, y - 30, last_chat["reply"]["doctor"])

    c.drawString(50, y - 60, "Advice:")
    c.drawString(70, y - 80, str(last_chat["reply"]["precaution"])[:90])

    c.drawString(50, y - 120, "Disclaimer:")
    c.drawString(70, y - 140, "This is guidance only and not a replacement for a certified doctor.")

    c.save()
    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
