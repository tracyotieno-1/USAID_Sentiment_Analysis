## USAID Sentiment Analysis
### Public Sentiment Analysis on USAID After Trump’s Withdrawal
### Business Understanding

The United States Agency for International Development (USAID) provides money for a range of initiatives, including those related to economic development, education, and health. However, the public's opinion of USAID has been deteriorating recently, particularly since President Donald Trump stopped supporting some of its initiatives. This has sparked debates on websites like Reddit and Youtube, where users share their differing views on the funding choices, policies, and effects of USAID. Policymakers and stakeholders need to understand public sentiment to evaluate how the public is responding to USAID programs. Governmental and non-governmental organizations to enhance communication tactics. Scholars examine how policy changes affect public opinion.

### Research Questions

- What is the Reddit and Youtube community's perception of USAID?
- Has the mood changed before and after significant policy choices (such as funding cuts)?
- Which USAID-related issues or compliments are most frequently expressed?

By using natural language processing (NLP) techniques to categorize sentiment (positive, negative, or neutral), this study seeks to analyze public opinion regarding USAID. It also aims to find important conversation topics, spot changes in sentiment patterns over time, and investigate how political events—like funding cuts—affect public opinion. The results will shed light on how USAID is viewed and guide future communication or policy initiatives.

### Problem Statement
Discussions about the impact of the USAID funding withdrawal are common on social media sites like Reddit and Youtube, where users share a range of viewpoints. These conversations are still unstructured, though, and challenging to do large-scale analysis. Policymakers, NGOs, and stakeholders must comprehend public opinion of USAID to evaluate how the organization's role and efficacy are perceived.

### Objectives
- Analyze public sentiment on USAID after Trump’s withdrawal.
- Extract key discussion themes from Reddit and Youtube posts.
- Build a chatbot for interactive sentiment queries.


### Success Metrics:
Model Accuracy: Achieve at least 85% accuracy for sentiment classification.
Business Insights: Identify 3-5 key themes in USAID discussions.
Engagement: Provide actionable insights for stakeholders (e.g., NGOs, government, researchers).

### Data Understanding
The project's dataset is made up of Reddit and Youtube comments about USAID that show how the public feels about its funding choices, policies, and overall effects. Key attributes like comment text, author, score (upvotes/downvotes), timestamp, and subreddit source are included in the CSV file "usaid_reddit_posts_New.csv". The data will be classified using sentiment analysis techniques unless there is a sentiment column that gives labeled sentiment categories (Positive, Negative, or Neutral). To guarantee data quality, preliminary investigation entails looking for missing values, duplicates, and class distribution. Analysis of word frequency and comment length also aids in identifying recurring themes in the conversations. Finding potential biases, data imbalances, and required preprocessing steps like text cleaning, stopword removal, and vectorization all depend on this step. Effective NLP modeling and sentiment analysis are predicated on a thorough comprehension of the dataset.


### LSTM and CNN models. 

These procedures were essential for preparing the data for deep learning models, cleaning it, and turning textual information into numerical representations.

Preprocessing Steps for Text The following changes were made before the text data was fed into the models:
Lowercasing: To maintain consistency, all text was converted to lowercase.
Removing Special Characters & Punctuation: To make the text cleaner, extraneous symbols were removed.
Tokenization: To improve processing, divide sentences into individual words.
Stopword Removal: Common words that don't help with sentiment understanding, like the, is, and, were eliminated.
Lemmatization: To keep things consistent, words are reduced to their root forms (running → run, for example).
Vectorization of Text Since text cannot be directly processed by deep learning models, the data was converted into numerical representations using.

A. Word embeddings, such as GloVe, Word2Vec, or TF-IDF

Word2Vec/GloVe: Converted words into contextually meaningful, dense vector representations.
Optional TF-IDF: Gives word importance scores according to how frequently they occur in the dataset.

B. Indexing and Padding of Tokens

Tokenization using Keras Tokenizer: Text was transformed into word index sequences.
Padding Sequences: By padding shorter sequences, a consistent input length was guaranteed.
Dividing Data to Train Models
Train-Test Split (80-20%): Training and testing sets were created from the dataset.
A portion of the training data was reserved for validation, resulting in a train-validation split of 80–20%.

### CNN Models: Normalization
`Feature Scaling:` To guarantee stable training, embeddings were scaled because CNNs benefit from normalized inputs.
Data Transformation's Effects

By eliminating noise and unnecessary data, the model's efficiency was increased.
Enhanced feature representation made it possible for CNN and LSTM to recognize significant patterns.
Reduced computational complexity by guaranteeing constant input length.
The dataset was optimized for deep learning-based sentiment classification through this methodical transformation process, guaranteeing high model accuracy and dependability.

### Model Performance Overview
To enable a data-driven understanding of public sentiment, policy impacts, and potential areas of concern, this project intends to extract meaningful insights from public discussions on USAID through sentiment analysis and predictive modeling.

`Long Short term memory (LSTM):` Metric performance: 87% accuracy F1-Score: 87%, 95% is the AUC.

#### Strengths:

Sequential Data Handling: The LSTM model is ideally suited for sentiment analysis since it successfully captures contextual dependencies in text.
High Predictive Power: The model exhibits strong discrimination between sentiment classes (positive, negative, and neutral), as evidenced by its 97% AUC score.
Balanced Performance: The high F1-score indicates that the model reduces the chance of misclassification by maintaining a good balance between precision and recall.

#### Weaknesses:

Computationally Expensive: Because LSTMs are sequential, they demand a large amount of computational resources and training time.
Possible Overfitting: Deeper LSTMs can occasionally memorize patterns, which can impact generalization on unseen data, even though this is not visible in the results as of yet.

Implication: The LSTM model performs exceptionally well in sentiment classification, which makes it a very dependable tool for figuring out public opinion and trends regarding issues on USAID.

The CNN, or convolutional neural network:
Metrics of Performance: 85% accuracy, F1-Score: 84%, 94% is the AUC.

#### Strengths:

Effective Feature Extraction: CNNs use convolutional filters to effectively extract important sentiment patterns from text data.
Comparable Performance: CNN is a powerful substitute for LSTM in sentiment analysis, achieving the same accuracy, F1-score, and AUC score.
Faster Training: Because CNNs can process information in parallel, they are computationally efficient and train more quickly than LSTMs.

#### Weaknesses:

Limited Contextual Understanding: CNNs may have trouble understanding long-range dependencies in text because they concentrate on local features.
Possible Information Loss: Convolution layers' fixed window sizes may cause some contextual meaning to be lost.
Implication: In situations where computational efficiency is a top concern, the CNN model is a great substitute for LSTM. Nevertheless, in contrast to sequential models, it might not have a deeper contextual understanding.


### Conclusion: 

- `Successful Sentiment Analysis Model` – We developed and deployed a sentiment analysis model that accurately predicts sentiment on USAID-related discussions while also providing word count insights for deeper text analysis.
- `Chatbot for USAID Discussions` – A specialized chatbot was built and deployed, designed to engage users exclusively on topics related to USAID, ensuring focused and informative interactions.
- `Data-Driven Insights for Stakeholders` – The sentiment analysis model provides valuable insights for policymakers, NGOs, and donors, helping them understand public perception and refine communication or funding strategies.
- `High Model Performance` – LSTM was chosen for sentiment classification due to its superior accuracy and F1 scores across all sentiment classes, ensuring reliable sentiment detection.
- `Future Enhancements` – The project can be expanded by incorporating real-time sentiment tracking, integrating more data sources, and improving the chatbot’s conversational depth using advanced NLP techniques.
