
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# -----------------------------------
# 1. Logistic Regression Model
# -----------------------------------

def create_logistic_model():

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    return model


# -----------------------------------
# 2. Naive Bayes Model
# -----------------------------------

def create_naive_bayes_model():

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

    return model


# -----------------------------------
# 3. Create Both Models
# -----------------------------------

def create_models():

    models = {
        "Logistic Regression": create_logistic_model(),
        "Naive Bayes": create_naive_bayes_model()
    }

    return models
