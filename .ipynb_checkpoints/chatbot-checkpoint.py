from flask import Flask, request, jsonify
import pandas as pd

# Load the sentiment-analyzed dataset
data_usaid = pd.read_csv("sentiment_usaid_data.csv")

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get("query", "").lower()
    
    # Check if the query contains "USAID"
    if "usaid" in user_query:
        # Filter relevant rows
        relevant_data = data_usaid[data_usaid['Cleaned_Text'].str.contains("usaid", case=False, na=False)]
        
        # If no results found, return a default message
        if relevant_data.empty:
            return jsonify({"response": "Sorry, I couldn't find anything related to USAID in the dataset."})
        
        # Return the first relevant result (or modify to return multiple results)
        response = relevant_data.iloc[0]['Cleaned_Text']
        return jsonify({"response": response})
    else:
        return jsonify({"response": "Sorry, that's beyond my current scope. Let’s talk about something else."})

if __name__ == "__main__":
    app.run(debug=True)
