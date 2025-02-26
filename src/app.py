import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# ======================= LSTM SENTIMENT ANALYSIS ======================= #

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Define LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Load trained LSTM model
input_size = 5000  # Adjust based on TF-IDF vectorizer features
hidden_size = 256
num_layers = 2
num_classes = 3
dropout_rate = 0.3

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_rate)
model.load_state_dict(torch.load("lstm_sentiment_model.pth"))
model.eval()

# Sentiment labels
sentiment_labels = ['Negative 😠', 'Neutral 😐', 'Positive 😊']

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    text_length = len(text)

    # Vectorize input text
    text_tfidf = vectorizer.transform([text]).toarray()
    text_tensor = torch.FloatTensor(text_tfidf).unsqueeze(1)  # Add batch dimension

    # Predict sentiment
    with torch.no_grad():
        output = model(text_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    sentiment = sentiment_labels[predicted_label]

    return jsonify({"sentiment": sentiment, "length": text_length})


# ========================= GEMINI AI CHATBOT ========================= #

# Configure Gemini API
API_KEY = "AIzaSyDX7D7QcNjZs8fGOe9Jia3XzTmbBmqsNiI"
genai.configure(api_key=API_KEY)

# Load Gemini model
chat_model = genai.GenerativeModel("gemini-1.5-flash")

# Load cleaned dataset (for chatbot context)
df = pd.read_csv("cleaned_data.csv")  # Ensure this contains 'cleaned_text' column

# Max context size
MAX_CONTEXT_SIZE = 10000

# Function to check if query is related to USAID
def is_related_to_usaid(query):
    keywords = ["USAID", "Trump", "Musk", "foreign aid", "development", "funding", "humanitarian"]
    return any(keyword.lower() in query.lower() for keyword in keywords)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_query = request.json.get("query", "").strip()

        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        if not is_related_to_usaid(user_query):
            return jsonify({"response": "Sorry, I can only discuss USAID-related topics."})

        # Ensure dataset column exists
        if "cleaned_text" not in df.columns:
            return jsonify({"error": "'cleaned_text' column missing"}), 500

        # Sample context data within MAX_CONTEXT_SIZE
        random_texts, total_length = [], 0
        sampled_df = df.sample(frac=1, random_state=random.randint(1, 100))  # Shuffle

        for text in sampled_df["cleaned_text"].astype(str).tolist():
            if total_length + len(text) > MAX_CONTEXT_SIZE:
                break
            random_texts.append(text)
            total_length += len(text)

        context = " ".join(random_texts)

        # Generate AI response
        response = chat_model.generate_content(f"Context: {context}\nUser Question: {user_query}")

        bot_response = response.text.strip() if response else "I'm sorry, I couldn't find relevant information."

        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================= FRONTEND & SERVER ========================= #

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
