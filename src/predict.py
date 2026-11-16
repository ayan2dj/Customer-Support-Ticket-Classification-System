
import re
import string
import joblib

from pathlib import Path


# -----------------------------------
# 1. Locate Saved Model
# -----------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "model" / "ticket_classifier.pkl"


# -----------------------------------
# 2. Load Trained Model
# -----------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train.py first."
    )

model = joblib.load(MODEL_PATH)


# -----------------------------------
# 3. Text Cleaning Function
# -----------------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# -----------------------------------
# 4. Predict Ticket Category
# -----------------------------------

def predict_ticket(description):

    if not description or not description.strip():
        raise ValueError(
            "Please enter a valid ticket description."
        )

    cleaned_description = clean_text(description)

    # Predict category
    prediction = model.predict(
        [cleaned_description]
    )[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(
        [cleaned_description]
    )[0]

    # Get confidence of predicted category
    confidence = max(probabilities) * 100

    return prediction, confidence


# -----------------------------------
# 5. Automatic Suggested Responses
# -----------------------------------

def generate_response(category):

    responses = {

        "Login Issue":
        "Please use the Forgot Password option "
        "on the login page to reset your password. "
        "If the problem continues, contact our support team.",

        "Application Error":
        "We apologize for the inconvenience. "
        "Please try restarting the application and "
        "check whether the issue persists. "
        "Our technical team can investigate further.",

        "Report":
        "Please navigate to the Reports section "
        "and select the required report type and date range. "
        "If you face any difficulty, contact our support team.",

        "Account Update":
        "Please visit your account profile settings "
        "to update your registered information. "
        "For verification-related changes, please contact support.",

        "Performance":
        "We apologize for the inconvenience. "
        "Please check your internet connection and "
        "try refreshing the application. "
        "Our technical team can investigate performance issues."

    }

    return responses.get(
        category,
        "Thank you for contacting us. "
        "Our support team will review your request."
    )
