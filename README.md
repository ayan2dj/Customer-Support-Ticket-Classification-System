The assignment specifically requires these ten sections:
1. Project overview
2. Problem statement
3. Technologies used
4. Dataset information
5. Installation steps
6. Required dependencies
7. How to train the model
8. How to perform prediction
9. How to run the application
10. Sample input/output

Here is a ready-to-use README draft.

## TECHNICAL PROJECT REPORT

# Customer Support Ticket Classification System Using Machine Learning

## 1. Problem Understanding

Customer support departments receive numerous requests through emails, chat systems and other communication channels. These requests need to be categorized so that they can be directed to the appropriate support team.

Manual classification can consume time and resources.

The objective of this project is to develop an AI/ML-based system that automatically predicts the category of a customer support ticket based on its textual description.

The system classifies tickets into five categories:

1. Login Issue
2. Application Error
3. Report
4. Account Update
5. Performance

## 2. Dataset Description

A synthetic customer support ticket dataset was generated using Python and Pandas.

### Dataset details

| Attribute | Value |
|---|---|
| Initial records | 250 |
| Number of categories | 5 |
| Initial records per category | 50 |
| Unique descriptions after cleaning and deduplication | 50 |
| File format | CSV |
| Dataset source | Self-generated synthetic dataset |

### Dataset fields

- **ticket_id:** Unique ticket identifier.
- **ticket_description:** Description of the customer's problem.
- **category:** Target classification label.
- **priority:** Ticket priority.
- **status:** Current ticket status.

The dataset was generated specifically for this assignment and does not contain real customer information.

After removing repeated descriptions, 50 unique records remained, with 10 records in each category.

## 3. Data Preprocessing

Data preprocessing was performed to improve data consistency and prepare ticket descriptions for machine learning.

The following operations were implemented:

### 3.1 Missing Value Handling

Missing values in essential fields such as ticket ID, ticket description and category were removed.

Missing priority and status values were replaced with "Unknown".

### 3.2 Duplicate Removal

Completely duplicated records and duplicate ticket IDs were removed.

Repeated cleaned ticket descriptions were also removed before model training to reduce the risk of identical descriptions appearing in both training and testing datasets.

### 3.3 Text Cleaning

The following text preprocessing techniques were implemented:

- Conversion of text into lowercase.
- Removal of URLs.
- Removal of punctuation.
- Removal of unnecessary whitespace.
- Normalization of ticket descriptions.

These operations helped maintain consistency in the textual data.

## 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed using Pandas, Matplotlib and Seaborn.

The following analyses and visualizations were generated:

1. Category distribution.
2. Priority distribution.
3. Status distribution.
4. Category versus priority distribution.

### Key observations

- The initial dataset contained 250 records.
- Each of the five categories contained 50 records.
- Low priority tickets: 85.
- Medium priority tickets: 85.
- High priority tickets: 80.
- Open tickets: 85.
- In Progress tickets: 85.
- Closed tickets: 80.

The initial dataset had a balanced distribution across categories, priorities and statuses.

After duplicate-description removal, the effective training dataset contained 50 unique descriptions.

## 5. Model Selection

A supervised machine-learning approach was selected for multiclass text classification.

### 5.1 Feature Extraction: TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) was used to convert textual ticket descriptions into numerical feature vectors.

TF-IDF assigns importance to words based on their frequency within a document and their distribution across the dataset.

Unigrams and bigrams were considered using:

`TfidfVectorizer(ngram_range=(1, 2))`

### 5.2 Classification Algorithms

Two machine-learning algorithms were implemented.

**A. Logistic Regression**

Logistic Regression estimates the probability of an input belonging to different categories and predicts the category with the highest estimated probability.

It was selected because it is computationally efficient and suitable for text classification using sparse TF-IDF features.

**B. Multinomial Naive Bayes**

Multinomial Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It was selected because it is computationally efficient and commonly used for text classification tasks.

Both algorithms were implemented using Scikit-learn Pipelines combining TF-IDF vectorization and classification.

## 6. Model Training and Testing

The dataset was divided into training and testing subsets using an 80:20 split.

### Training configuration

| Parameter | Value |
|---|---|
| Initial dataset records | 250 |
| Unique descriptions | 50 |
| Training records | 40 |
| Testing records | 10 |
| Training percentage | 80% |
| Testing percentage | 20% |
| Random state | 42 |
| Stratification | Category |

Stratified splitting was used to maintain equal category representation in the training and testing subsets.

Both Logistic Regression and Multinomial Naive Bayes were trained using the training dataset and evaluated using the held-out testing dataset.

## 7. Model Evaluation and Results

Both models were evaluated using accuracy, precision, recall and F1-score.

### 7.1 Performance Comparison

The following results were obtained from the actual execution of the training program.

| Evaluation Metric | Logistic Regression | Naive Bayes |
|---|---:|---:|
| Accuracy | 70.00% | 70.00% |
| Precision (Macro) | 63.33% | 63.33% |
| Recall (Macro) | 70.00% | 70.00% |
| F1-score (Macro) | 65.67% | 65.67% |

### 7.2 Interpretation of Results

**Accuracy: 70%**

Both models correctly classified 7 out of 10 testing records.

**Precision: 63.33%**

The macro-averaged precision indicates that the average proportion of correct predictions among predicted examples across the five categories was approximately 63.33%.

**Recall: 70%**

The macro-averaged recall indicates that the models correctly identified approximately 70% of the actual examples across the five categories.

**F1-score: 65.67%**

The macro F1-score represents the balance between precision and recall across the five categories.

### 7.3 Classification Report

The following class-wise results were obtained from the model evaluation:

| Category | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Account Update | 1.00 | 0.50 | 0.67 | 2 |
| Application Error | 0.50 | 1.00 | 0.67 | 2 |
| Login Issue | 1.00 | 1.00 | 1.00 | 2 |
| Performance | 0.00 | 0.00 | 0.00 | 2 |
| Report | 0.67 | 1.00 | 0.80 | 2 |
| **Macro Average** | **0.63** | **0.70** | **0.63** | **10** |

Note: The rounded macro F1-score in the classification report is 0.63, while the separately printed F1-score is 0.6567. The latter is the value reported by the training program's `f1_score` calculation.

### 7.4 Confusion Matrix

A confusion matrix was generated to visualize the relationship between actual and predicted categories.

The matrix shows correct classifications along the diagonal and incorrect classifications outside the diagonal.

**[Insert screenshots/confusion_matrix.png here]**

### 7.5 Model Selection

Both algorithms achieved identical evaluation results:

- Accuracy: 70%.
- Macro Precision: 63.33%.
- Macro Recall: 70%.
- Macro F1-score: 65.67%.

Therefore, neither algorithm demonstrated superior predictive performance on this particular test set.

The implementation uses Logistic Regression as the selected model when the F1-scores are tied because it appears first in the model dictionary.

## 8. New Ticket Prediction

The trained classification system was tested using five new ticket descriptions that were not part of the training records.

### Actual prediction results

| No. | Test Input | Expected Category | Predicted Category | Confidence |
|---|---|---|---|---:|
| 1 | My sign-in credentials are not being accepted. | Login Issue | Login Issue | 22.63% |
| 2 | The software crashes whenever I open settings. | Application Error | Application Error | 27.88% |
| 3 | I need an export of last month's revenue. | Report | Report | 25.74% |
| 4 | Please change the email linked to my profile. | Account Update | Account Update | 30.18% |
| 5 | The dashboard takes several minutes to respond. | Performance | Performance | 26.71% |

### Prediction Analysis

All five test descriptions were classified into their expected categories.

The predicted confidence scores ranged from 22.63% to 30.18%.

Although the predicted categories were correct, the confidence scores were relatively low. This indicates that the model does not strongly distinguish the selected categories for these examples.

These predictions demonstrate basic classification functionality but do not establish reliable real-world performance.

**[Insert screenshots/prediction.png here]**

## 9. Simple User Interface

A web-based interface was developed using Streamlit.

The application allows users to enter customer support ticket descriptions and receive automated classification results.

### Main features

- Customer ticket description input.
- Predict Category button.
- Predicted ticket category.
- Model confidence percentage.
- Progress bar for confidence visualization.
- Suggested customer response.

The application uses the saved machine-learning pipeline to make predictions without retraining the model for every request.

The suggested responses are generated using predefined rule-based messages associated with each category.

**[Insert screenshots/web_interface.png here]**

## 10. Limitations

The current implementation has the following limitations:

1. The dataset is synthetically generated rather than collected from real customer interactions.
2. Only 50 unique descriptions remain after duplicate removal.
3. The testing dataset contains only 10 records, making the evaluation statistically limited.
4. The system supports only five predefined ticket categories.
5. Similar or ambiguous ticket descriptions may result in incorrect classifications.
6. The model's estimated confidence scores are relatively low and have not been calibrated.
7. Suggested customer responses are predefined and may not fully resolve every customer issue.

## 11. Future Improvements

The system can be improved in several ways:

- Increase the dataset size using diverse real-world customer support examples.
- Introduce additional ticket categories.
- Experiment with advanced NLP and deep-learning models.
- Improve model performance through cross-validation and hyperparameter tuning.
- Calibrate probability estimates to improve confidence interpretation.
- Introduce human feedback for incorrect predictions.
- Integrate a database for ticket management.
- Develop a production-ready REST API.
- Introduce authentication and role-based access.
- Use Large Language Models for more flexible customer responses.

## 12. Conclusion

This project demonstrates the development of a basic machine-learning-based customer support ticket classification system.

The system uses text preprocessing, TF-IDF feature extraction and supervised classification algorithms to predict customer support ticket categories.

Logistic Regression and Multinomial Naive Bayes were implemented and compared using standard evaluation metrics.

Both models achieved 70% accuracy and a macro F1-score of 65.67% on the small held-out test dataset.

The system was also tested with five new ticket descriptions, and all five were assigned their expected categories.

A Streamlit-based web interface was developed to allow users to submit ticket descriptions and view predicted categories, confidence scores and suggested responses.

The project provides practical experience in Python programming, data processing, natural language processing, machine learning, model evaluation and application development.

## 13. Tools and Technologies

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Joblib

## 14. AI Tools Disclosure

AI tools were used for conceptual explanations, debugging assistance, implementation guidance and documentation preparation.

The implementation and generated results were reviewed and tested during project development.

The developer is responsible for understanding and explaining the submitted code and implementation decisions.