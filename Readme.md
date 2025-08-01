# 💊 AI Drug Interaction & Side Effect Prediction System

An AI-driven application that analyzes potential **drug interactions** and predicts **adverse side effects**, aiming to assist healthcare professionals in making safer and more informed medication decisions.

---

## 🚀 Overview

This project leverages:
- **RAG (Retrieval-Augmented Generation)** for generating contextual insights.
- **MiniLM-L6-v2** model for efficient language representation.
- **FAISS** and **Vector Databases** for semantic search and similarity comparison.
- **Flask** for building a lightweight web application interface.

---

## 📁 Project Structure

```bash
AI DRUG/
│
├── faiss_index/
├── my_env/
├── static/
├── templates/
├── uploads/
├── venv/
│
├── .gitignore
├── app.py
│
├── FAERS_Dataset.csv
├── final drug dataset.csv
├── final_varied_drug_inter.csv
│
├── label_encoder_condition.pkl
├── label_encoder_drug.pkl
├── label_encoder_effect.pkl
├── scaler.pkl
│
├── model1.ipynb
├── model2.ipynb
├── ragmodel.ipynb
├── ragmodel.pkl
├── Long term drug.ipynb
```

---

## ⚙️ Key Features

- **Drug Interaction Risk Prediction:** Identifies and scores risks from drug combinations.
- **Side Effect Analysis:** Predicts side effects based on real-world reports and model inference.
- **RAG for Explanation:** Uses retrieval-augmented generation to produce contextual recommendations.
- **Semantic Search:** FAISS index enables vector-based retrieval of relevant drug information.
- **Web Interface:** Flask app for user-friendly interaction with model predictions.

---

## 📊 Datasets Used

- `FAERS_Dataset.csv`: FDA Adverse Event Reporting System dataset.
- `final drug dataset.csv`: Cleaned and preprocessed drug data.
- `final_varied_drug_inter.csv`: Enriched interaction dataset for training and evaluation.

---

## 🧠 Models & Tools

- **MiniLM-L6-v2**: Lightweight transformer for embedding drug descriptions.
- **Custom ML Models**: Trained to classify conditions, side effects, and interaction severity.
- **FAISS**: Vector similarity search engine for nearest neighbor retrieval.
- **RAG Pipeline**: Combines document retrieval with generative response to explain predictions.

---

## 📦 Dependencies

- `flask`
- `faiss-cpu`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`

---

## 📌 To-Do

- ✅ Predict drug interaction risks  
- ✅ Forecast likely side effects  
- ⬜ Add user login system  
- ⬜ Connect with real-time external drug knowledge bases (e.g., DrugBank, RxNorm)

---
