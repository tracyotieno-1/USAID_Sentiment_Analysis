### USAID Sentiment Analysis
### Public Sentiment Analysis on USAID After Trump’s Withdrawal
1. Project Overview
This project conducts sentiment analysis on Reddit discussions related to USAID funding. Using Natural Language Processing (NLP) techniques, it analyzes public sentiment in response to funding changes. The goal is to identify sentiment trends, correlations, and potential insights regarding USAID’s perception over time.
2. Business Understanding
The United States Agency for International Development (USAID) funds various initiatives, including economic development, education, and health. However, public perception of USAID has shifted, particularly since President Donald Trump withdrew funding for certain programs. This has sparked debates on platforms like Reddit, where users express varying opinions about funding decisions, policies, and USAID's overall impact.
Understanding public sentiment is crucial for policymakers, NGOs, and analysts to evaluate responses to USAID programs. The key questions this project seeks to answer include:
•	What is the general perception of USAID on Reddit?
•	How has sentiment evolved before and after significant policy decisions (such as funding cuts)?
•	What are the most common concerns or praises about USAID?
•	Can sentiment trends predict public support for future policies?
3. Problem Statement
Discussions about USAID funding withdrawals are prevalent on social media, but these conversations are unstructured and difficult to analyze at scale. Policymakers, NGOs, and stakeholders require a structured understanding of public opinion to assess how USAID's role and effectiveness are perceived.
4. Objectives
1.	Analyze public sentiment on USAID after Trump’s withdrawal.
2.	Extract key discussion themes from Reddit posts.
3.	Compare sentiment across platforms (if other data sources are added later).
4.	Build a chatbot for interactive sentiment queries.
5. Success Metrics
•	Model Accuracy: Achieve at least 85% accuracy for sentiment classification.
•	Business Insights: Identify 3-5 key themes in USAID discussions.
•	Engagement: Provide actionable insights for stakeholders (e.g., NGOs, government, researchers).
6. Dataset Used
The project utilizes the following dataset:
•	Reddit USAID Comments Dataset: Contains posts, titles, and comments related to USAID funding.
•	The text fields are merged into a single "merged text" column for analysis.
•	Cleaned dataset saved as cleaned_dataset.csv after preprocessing.
7. Data Understanding & Preprocessing
7.1 Data Collection
•	Extracted posts and comments from relevant subreddits using the Reddit API (PRAW).
•	Captured comment text, timestamps, user metadata, and engagement metrics (upvotes/downvotes).
•	Ensured diverse representation of opinions on USAID policies and funding.
7.2 Data Cleaning
•	Removed Missing Values: Eliminated null or incomplete entries.
•	Merged Text Fields: Combined multiple text columns into a single column for analysis.
•	Tokenization: Split text into individual words.
•	Lowercasing: Standardized text formatting.
•	Stopword Removal: Removed common words that do not add meaningful context.
•	Lemmatization: Reduced words to their root form for consistency.
•	Exported Cleaned Data: df.to_csv('cleaned_dataset.csv', index=False)
8. Exploratory Data Analysis (EDA)
Key Visualizations:
(Place visualizations here)
•	Sentiment Distribution: Analyzed the proportion of positive, negative, and neutral comments.
•	Word Clouds: Highlighted the most frequently used words in different sentiment categories.
•	Time-Series Analysis: Tracked changes in sentiment over time.
•	Top Keywords: Identified commonly used words in discussions about USAID.
9. Sentiment Analysis Model
9.1 Model Selection
The project experimented deep learning models:
1.	Deep Learning Models: 
o	LSTM (Recurrent Neural Network): Captures contextual dependencies.
o	CNN (Convolutional Neural Network): Extracts key text features efficiently.
9.2 Feature Engineering
•	TF-IDF Vectorization: Captured word importance in the dataset.
•	Word Embeddings (GloVe/Word2Vec): Transformed words into dense vector representations.
•	Padding & Normalization: Ensured consistent input length for deep learning models.
9.3 Model Evaluation
LSTM Model:
•	Accuracy: 85%
•	F1-Score: 84%
•	AUC: 94%
Strengths:
•	Handles sequential data well
•	High predictive power and contextual understanding
Weaknesses:
•	Computationally expensive
•	Potential for overfitting
CNN Model:
•	Accuracy: 85%
•	F1-Score: 84%
•	AUC: 94%
Strengths:
•	Faster training time
•	Effective feature extraction
Weaknesses:
•	Limited ability to understand long-range dependencies
10. Handling Class Imbalance
•	Applied SMOTE (Synthetic Minority Over-Sampling Technique) to balance sentiment classes.
11. Model Deployment
•	Developed a chatbot to classify new comments dynamically.
•	Integrated model for real-time sentiment analysis.
12. Insights
1.	Public Perception is Highly Contextual: External events significantly impact sentiment trends.
2.	Neutral Sentiment Dominates: Many Reddit users discuss USAID factually rather than emotionally.
3.	Deep Learning Models Improve Accuracy: LSTM performed best in capturing text patterns.
13. Future Improvements
1.	Expand Data Sources: Include Twitter and news articles.
2.	Use Transformer Models: Implement BERT or GPT for better text understanding.
3.	Enhance Chatbot Deployment: Develop a web-based interactive sentiment analysis tool.
14. ### Conclusion
Both CNN and LSTM models performed well, achieving 94% AUC, 84% F1-score, and 85% accuracy. CNN offers faster training, while LSTM provides superior contextual understanding. The study provides valuable insights into public sentiment on USAID, informing policymakers and NGOs about public discourse trends.
