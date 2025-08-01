from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import joblib
import pandas as pd
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
import faiss
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# File Upload Config
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load FAISS Index & Metadata
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
index = faiss.read_index("faiss_index/index.faiss")

with open("faiss_index/index.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load ML Models
model = joblib.load("ragmodel.pkl")
scaler = joblib.load("scaler.pkl")
drug_encoder = joblib.load("label_encoder_drug.pkl")
condition_encoder = joblib.load("label_encoder_condition.pkl")
side_effect_encoder = joblib.load("label_encoder_effect.pkl")

# --------- USER AUTHENTICATION ---------
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id == "1" else None

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == "admin" and password == "password":
            user = User(user_id="1")
            login_user(user, remember=True)
            session.permanent = True
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index1'))
        
        flash("Invalid credentials. Try again.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

# --------- MAIN PAGES ---------
@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    retrieved_info = None  # Initialize variable

    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file uploaded!", "warning")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No file selected!", "warning")
            return redirect(request.url)

        if file and file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            extracted_text = extract_text_from_pdf(filepath)
            retrieved_info = retrieve_doctor_notes_from_pdf(extracted_text)

            flash("File uploaded and processed successfully!", "success")

        else:
            flash("Invalid file format. Please upload a PDF.", "danger")

    return render_template('upload.html', output=retrieved_info)

@app.route('/risk')
def risk():
    return render_template('risk.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# --------- RISK PREDICTION ---------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form.get('age'))
        gender = 1 if request.form.get('gender').lower() == "male" else 0  
        drug_name = request.form.get('drug')
        dosage = float(request.form.get('dosage'))
        duration_months = int(request.form.get('duration'))
        chronic_conditions = request.form.get('chronic')
        kidney_function = float(request.form.get('kidney'))
        creatinine_levels = float(request.form.get('creatinine'))
        liver_function = float(request.form.get('liver'))
        blood_pressure = int(request.form.get('blood'))
        blood_glucose = float(request.form.get('glucose'))
        side_effect = request.form.get('side_effect')

        drug_name_encoded = drug_encoder.transform([drug_name])[0] if drug_name in drug_encoder.classes_ else -1
        chronic_conditions_encoded = condition_encoder.transform([chronic_conditions])[0] if chronic_conditions in condition_encoder.classes_ else -1
        side_effect_encoded = side_effect_encoder.transform([side_effect])[0] if side_effect in side_effect_encoder.classes_ else -1

        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Drug Name": drug_name_encoded,
            "Dosage (mg/day)": dosage,
            "Duration (Months)": duration_months,
            "Chronic Conditions": chronic_conditions_encoded,
            "Kidney Function (GFR)": kidney_function,
            "Creatinine Levels (mg/dL)": creatinine_levels,
            "Liver Function (ALT/AST)": liver_function,
            "Blood Pressure (Systolic)": blood_pressure,
            "Blood Glucose (mg/dL)": blood_glucose,
            "Side Effect": side_effect_encoded
        }])

        input_scaled = scaler.transform(input_data)

        risk_severity = model.predict(input_scaled)[0]

        return render_template('result.html', risk_severity=risk_severity)

    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/error')
def error_page():
    error_msg = request.args.get('error_msg', "An unknown error occurred.")
    return render_template('error.html', error_msg=error_msg)

# --------- HELPER FUNCTIONS ---------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # ✅ Use 'fitz' instead of 'pymupdf'
    text = "\n".join([page.get_text("text") for page in doc])
    return text


def retrieve_doctor_notes_from_pdf(extracted_text, top_k=1):
    query_embedding = embedding_model.embed_query(extracted_text)
    search_results = vector_store.similarity_search_by_vector(query_embedding, k=top_k)
    
    return [
        {"doctor_notes": res.page_content, **res.metadata}
        for res in search_results
    ]

# Run Flask App on Port 5001
if __name__ == '__main__':
    app.run(debug=True, port=5001)
