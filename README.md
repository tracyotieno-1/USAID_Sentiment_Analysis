### USAID Sentiment Analysis
### Public Sentiment Analysis on USAID After Trump’s Withdrawal
### Business Understanding
The United States Agency for International Development (USAID) funds various initiatives, including economic development, education, and healthcare. However, public perception of USAID has declined, particularly following President Donald Trump's decision to withdraw funding from some of its programs. This has ignited discussions on platforms like Reddit, where user’s express diverse opinions on funding decisions, policies, and USAID's overall impact. Understanding public sentiment is crucial for policymakers, stakeholders, and governmental and non-governmental organizations to refine communication strategies and evaluate the effects of policy changes.
Key questions to consider:
•	What is the general sentiment of the Reddit community regarding USAID?
•	Has sentiment shifted before and after key policy decisions (e.g., funding cuts)?
•	What are the most frequently discussed issues or praises related to USAID?
•	Can sentiment trends be used to predict public support for future policies?

### Problem Statement
While discussions about the impact of USAID funding withdrawal are widespread on social media platforms like Reddit, these conversations remain unstructured and difficult to analyze at scale. Policymakers, NGOs, and other stakeholders must understand public opinion on USAID to assess perceptions of its role and effectiveness.

### Objectives
•	Analyze public sentiment toward USAID following Trump's funding withdrawal.
•	Identify key themes in Reddit discussions about USAID.
•	Compare sentiment across platforms if additional data sources become available.
•	Develop a chatbot for interactive sentiment analysis queries.

### Project Goals
This research utilizes Natural Language Processing (NLP) to analyze Reddit comments about USAID, identifying sentiment trends and discussion topics.
Business Impact
•	Provide insights into policy effectiveness and areas of concern.
•	Assist NGOs and policymakers in understanding public sentiment regarding USAID.
•	Support academics, analysts, and journalists in monitoring discussion patterns.

### Success Metrics
•	Model Accuracy: Achieve at least 85% accuracy in sentiment classification.
•	Business Insights: Identify 3-5 key themes in USAID discussions.
•	Engagement: Generate actionable insights for stakeholders, including NGOs, government agencies, and researchers.

### Data Understanding
The dataset consists of Reddit comments discussing USAID, capturing public sentiment regarding funding decisions, policies, and overall impact. The dataset, contained in usaid_reddit_posts_New.csv, includes key attributes such as comment text, author, score (upvotes/downvotes), timestamp, and subreddit source. Sentiment labels (Positive, Negative, Neutral) are either provided or inferred using sentiment analysis models.
Initial data exploration includes:
•	Checking for missing values, duplicates, and class distribution.
•	Analyzing word frequency and comment length to identify recurring themes.
•	Detecting potential biases, data imbalances, and required preprocessing steps such as text cleaning, stopword removal, and vectorization.

### Research Question
"What are the most commonly expressed opinions about USAID on Reddit, and how have these opinions evolved since funding was withdrawn?"
Using NLP techniques, this study aims to:
•	Categorize sentiment (positive, negative, or neutral).
•	Identify major discussion themes.
•	Track sentiment shifts over time.
•	Examine the impact of political events, such as funding cuts, on public perception.
Findings will offer insights into USAID's public perception and inform future communication or policy strategies.

### Data Transformation Overview
Several preprocessing and transformation steps were applied to Reddit comments to optimize model performance:
Text Preprocessing:

•	Lowercasing: Standardized all text to lowercase.
•	Removing Special Characters & Punctuation: Cleaned text for better model performance.
•	Tokenization: Split sentences into individual words.
•	Stopword Removal: Removed common words that do not contribute to sentiment understanding.
•	Lemmatization: Reduced words to their root forms (e.g., "running" → "run").
Text Vectorization:

•	Word Embeddings (GloVe, Word2Vec, or TF-IDF): Converted words into meaningful numerical representations.
•	Token Indexing & Padding: Ensured uniform input length for models.
Data Splitting & Normalization:

•	Train-Test Split (80-20%)
•	Train-Validation Split (80-20%)
•	Feature Scaling: Normalized embeddings for CNN models.
These steps optimized the dataset for deep learning-based sentiment classification, ensuring high accuracy and reliability.
### Data Collection Summary
The dataset was gathered from Reddit discussions on USAID, focusing on user-generated comments. Web scraping and the Reddit API (PRAW) were used to extract posts and comments from relevant subreddits. The data was cleaned, organized, and prepared for sentiment classification.
### Data Description
Key dataset attributes include:
•	Comment Text: Reddit comment expressing opinions on USAID.
•	Sentiment Label: Positive, Negative, or Neutral.
•	Timestamp: Date and time of comment posting.
•	Upvotes/Downvotes: Indicator of comment popularity.
•	Comment Length: Number of words or characters, potentially reflecting sentiment strength.
### Data Quality Assessment
•	Missing Values: Checked and handled missing sentiment labels and comment text.
•	Duplicates: Removed to prevent bias.
•	Text Cleaning: Eliminated noise such as URLs, special characters, and stopwords.
•	Class Imbalance: Addressed through oversampling, undersampling, or weighted loss functions.
•	Spelling & Grammar Issues: Normalized text for consistency.
### Data Visualization Summary
Key visualizations include:
•	Sentiment Distribution Plot: Identifies class balance or skewness.
•	Word Clouds: Highlights frequently used terms by sentiment category.
•	Bar Charts: Displays top words influencing sentiment classification.
•	Sentiment Trend Analysis: Tracks sentiment shifts over time in response to major events.

### Model Performance Overview
This project applies sentiment analysis and predictive modeling to extract meaningful insights from public discussions on USAID.
Recurrent Neural Network (RNN-LSTM)
Performance Metrics: 85% Accuracy, F1-Score: 84%, AUC: 94%
Strengths:

•	Sequential Data Handling: LSTM effectively captures contextual dependencies in text.
•	High Predictive Power: Strong discrimination between sentiment classes (Positive, Negative, Neutral), evidenced by a 94% AUC score.
•	Balanced Performance: The high F1-score suggests a strong balance between precision and recall, minimizing misclassifications.
Weaknesses:

•	Computationally Expensive: LSTMs require significant computational resources and training time.
•	Potential Overfitting: Deep LSTMs may memorize patterns, affecting generalization on unseen data.
Implication: The LSTM model demonstrates exceptional sentiment classification performance, making it a highly reliable tool for analyzing public opinion trends regarding USAID.
Convolutional Neural Network (CNN)
Performance Metrics: 85% Accuracy, F1-Score: 84%, AUC: 94%
Strengths:

•	Efficient Feature Extraction: Convolutional filters effectively identify key sentiment patterns.
•	Comparable Performance: CNN achieves similar accuracy, F1-score, and AUC as LSTM.
•	Faster Training: CNNs process data in parallel, making them computationally efficient.
Weaknesses:

•	Limited Contextual Understanding: CNNs focus on local features, potentially missing long-range dependencies.
•	Possible Information Loss: Fixed window sizes in convolution layers may lead to a loss of contextual meaning.
Implication: CNN serves as a robust alternative to LSTM, particularly in scenarios where computational efficiency is a priority.
### Conclusion: 
Both CNN and LSTM models achieve high performance, with 85% accuracy, 84% F1-score, and 94% AUC. While CNN is faster, LSTM captures deeper contextual dependencies. Either model can be used based on computational constraints and contextual requirements.
