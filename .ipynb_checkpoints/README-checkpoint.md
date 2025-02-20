### USAID Sentiment Analysis
### Public Sentiment Analysis on USAID After Trump’s Withdrawal
### Business Understanding
The United States Agency for International Development (USAID) provides money for a range of initiatives, including those related to economic development, education, and health. However, the public's opinion of USAID has been deteriorating recently, particularly since President Donald Trump stopped supporting some of its initiatives. This has sparked debates on websites like Reddit, where users share their differing views on the funding choices, policies, and effects of USAID. Policymakers and stakeholders need to understand public sentiment to evaluate how the public is responding to USAID programs. Governmental and non-governmental organizations to enhance communication tactics. Scholars examine how policy changes affect public opinion.

Among the issues we must ask ourselves are:

What is the Reddit community's perception of USAID?
Has the mood changed before and after significant policy choices (such as funding cuts)?
Which USAID-related issues or compliments are most frequently expressed?
Is it possible to forecast public support for future policies using sentiment trends?

### Problem Statement
Discussions about the impact of the USAID funding withdrawal are common on social media sites like Reddit, where users share a range of viewpoints. These conversations are still unstructured, though, and challenging to do large-scale analysis. Policymakers, NGOs, and stakeholders must comprehend public opinion of USAID to evaluate how the organization's role and efficacy are perceived.

### Objectives
Analyze public sentiment on USAID after Trump’s withdrawal.
Extract key discussion themes from Reddit posts.
Compare sentiment across platforms (if other data sources are added later).
Build a chatbot for interactive sentiment queries.

### Project Goals
This research aims to use Natural Language Processing (NLP) to examine Reddit comments about USAID and identify sentiment patterns and important subjects.

##Business Impact:

Offer insights into the efficacy of policies and areas of concern.
Assist NGOs and policymakers in understanding public opinion about USAID.
Help academics, analysts, and journalists monitor patterns in discourse.

### Success Metrics:
Model Accuracy: Achieve at least 85% accuracy for sentiment classification.
Business Insights: Identify 3-5 key themes in USAID discussions.
Engagement: Provide actionable insights for stakeholders (e.g., NGOs, government, researchers).

### Data Understanding
The project's dataset is made up of Reddit comments about USAID that show how the public feels about its funding choices, policies, and overall effects. Key attributes like comment text, author, score (upvotes/downvotes), timestamp, and subreddit source are included in the CSV file "usaid_reddit_posts_New.csv". The data will be classified using sentiment analysis techniques unless there is a sentiment column that gives labeled sentiment categories (Positive, Negative, or Neutral). To guarantee data quality, preliminary investigation entails looking for missing values, duplicates, and class distribution. Analysis of word frequency and comment length also aids in identifying recurring themes in the conversations. Finding potential biases, data imbalances, and required preprocessing steps like text cleaning, stopword removal, and vectorization all depend on this step. Effective NLP modeling and sentiment analysis are predicated on a thorough comprehension of the dataset.

### Research Question
"What opinions about USAID are most commonly voiced on Reddit, and how have these opinions changed since funding was cut off?"

By using natural language processing (NLP) techniques to categorize sentiment (positive, negative, or neutral), this study seeks to analyze public opinion regarding USAID. It also aims to find important conversation topics, spot changes in sentiment patterns over time, and investigate how political events—like funding cuts—affect public opinion. The results will shed light on how USAID is viewed and guide future communication or policy initiatives.

### Model Performance Overview
To enable a data-driven understanding of public sentiment, policy impacts, and potential areas of concern, this project intends to extract meaningful insights from public discussions on USAID through sentiment analysis and predictive modeling.

Recurrent Neural Networks (RNN-LSTM):
Metric performance: 85% accuracy F1-Score: 84%, 94% is the AUC.

Strengths:

Sequential Data Handling: The LSTM model is ideally suited for sentiment analysis since it successfully captures contextual dependencies in text.
High Predictive Power: The model exhibits strong discrimination between sentiment classes (positive, negative, and neutral), as evidenced by its 94% AUC score.
Balanced Performance: The high F1-score indicates that the model reduces the chance of misclassification by maintaining a good balance between precision and recall.

Weaknesses:

Computationally Expensive: Because LSTMs are sequential, they demand a large amount of computational resources and training time.
Possible Overfitting: Deeper LSTMs can occasionally memorize patterns, which can impact generalization on unseen data, even though this is not visible in the results as of yet.

Implication: The LSTM model performs exceptionally well in sentiment classification, which makes it a very dependable tool for figuring out public opinion and trends regarding issues on USAID.

The CNN, or convolutional neural network:
Metrics of Performance: 85% accuracy, F1-Score: 84%, 94% is the AUC.

Strengths:

Effective Feature Extraction: CNNs use convolutional filters to effectively extract important sentiment patterns from text data.
Comparable Performance: CNN is a powerful substitute for LSTM in sentiment analysis, achieving the same accuracy, F1-score, and AUC score.
Faster Training: Because CNNs can process information in parallel, they are computationally efficient and train more quickly than LSTMs.

Weaknesses:

Limited Contextual Understanding: CNNs may have trouble understanding long-range dependencies in text because they concentrate on local features.
Possible Information Loss: Convolution layers' fixed window sizes may cause some contextual meaning to be lost.
Implication: In situations where computational efficiency is a top concern, the CNN model is a great substitute for LSTM. Nevertheless, in contrast to sequential models, it might not have a deeper contextual understanding.

Conclusion

Both CNN and LSTM models perform well, achieving 94% AUC, 84% F1-score, and 85% accuracy. CNN provides faster training and competitive results, but LSTM is better at capturing long-term dependencies in text.

Based on the outcomes, both models can be applied interchangeably, contingent on the requirements for deep contextual understanding and computational limitations.

### Data Transformation Overview
Several preprocessing and transformation procedures were applied to the raw Reddit comments on USAID to guarantee the best possible performance of the:

LSTM and CNN models. 

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

CNN Models: Normalization
Feature Scaling: To guarantee stable training, embeddings were scaled because CNNs benefit from normalized inputs.
Data Transformation's Effects

By eliminating noise and unnecessary data, the model's efficiency was increased.
Enhanced feature representation made it possible for CNN and LSTM to recognize significant patterns.
Reduced computational complexity by guaranteeing constant input length.
The dataset was optimized for deep learning-based sentiment classification through this methodical transformation process, guaranteeing high model accuracy and dependability.

### Data Collection Summary
The project's dataset was gathered from Reddit discussions about USAID, with an emphasis on user-generated comments to gauge public opinion. To extract posts and comments from pertinent subreddits, web scraping techniques were used with the Reddit API (PRAW). Textual content, timestamps, user metadata, and upvote/downvote counts are among the information gathered. Several threads addressing USAID's funding, policies, and impact were used to collect comments to guarantee a varied representation of viewpoints. To prepare it for exploratory data analysis and sentiment classification, the data was subsequently cleaned and organized, eliminating redundant entries and unnecessary content.

### Data Description
Text-based Reddit comments about USAID make up the dataset, along with metadata that gives each comment context. Among the best attributes are:

Comment Text – The actual Reddit comment expressing opinions about USAID.
Sentiment Label – A categorical label indicating whether the sentiment of the comment is positive, negative, or neutral.
Timestamp – The date and time when the comment was posted, useful for analyzing sentiment trends over time.
Upvotes/Downvotes – The number of upvotes and downvotes a comment received, which can indicate its popularity or agreement within the community.
Comment Length – The number of words or characters in each comment, which may provide insights into sentiment strength.
To ensure consistency and enhance model performance, the dataset was preprocessed to clean and standardize the text.


### Data Quality Assessment.
A comprehensive data quality assessment was carried out to guarantee the dataset's dependability for sentiment analysis. Among the important factors assessed are:

Missing Values: Important fields like sentiment labels and comment text were examined for missing or null values in the dataset. Any entries that were missing were either eliminated or, if required, imputed.
Duplicates: To avoid bias in sentiment analysis, duplicate comments were found and eliminated.
Text Cleaning: To enhance the performance of the NLP model, the raw text data was cleaned of noise, including stopwords, special characters, URLs, and excessive whitespace.
Class Imbalance: To ascertain whether the dataset was biased toward a specific sentiment, the sentiment label distribution was examined. Techniques like oversampling, undersampling, or weighted loss functions were taken into consideration if an imbalance was discovered.
Spelling and Grammar Issues – Informal language and misspellings in Reddit comments were addressed using text normalization techniques to enhance data consistency.

### Data Visualization Summary
Several visualizations were carried out to gain a better understanding of the dataset's sentiment distribution and important patterns. To determine whether the dataset is balanced or skewed towards a specific sentiment, a count plot was used to create a sentiment distribution plot. Word clouds highlighting the most commonly used terms in each category were created for positive, negative, and neutral sentiments to investigate text patterns. Key terms influencing each sentiment classification were also revealed by bar charts showing the most frequently used words for each sentiment. To identify possible sentiment shifts brought on by significant events, a sentiment trend analysis over time was conducted to look at how opinions changed. Together, these visualizations provided a more in-depth understanding of the dataset, which aided in feature engineering and model training.