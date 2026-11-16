
import streamlit as st
import sys
from pathlib import Path

# =====================================
# 1. PROJECT PATH CONFIGURATION
# =====================================

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

# Add src directory to Python module search path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =====================================
# 2. IMPORT PREDICTION FUNCTIONS
# =====================================

try:
    from predict import predict_ticket, generate_response

except ImportError as e:
    st.error(f"Unable to import prediction module: {e}")
    st.stop()


# =====================================
# 3. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Customer Support Classifier",
    page_icon="🎫",
    layout="centered"
)


# =====================================
# 4. APPLICATION HEADER
# =====================================

st.title("🎫 Customer Support Classifier")

st.write(
    "An AI/ML-powered application that automatically "
    "classifies customer support tickets."
)

st.caption(
    "Built using Python, NLP, TF-IDF and Machine Learning."
)

st.divider()


# =====================================
# 5. TICKET DESCRIPTION INPUT
# =====================================

st.subheader("Enter Customer Ticket")

description = st.text_area(
    "Ticket Description",
    placeholder=(
        "Example: I cannot log in because "
        "my password is not working."
    ),
    height=150
)


# =====================================
# 6. PREDICTION BUTTON
# =====================================

if st.button(
    "🔍 Predict Category",
    type="primary",
    use_container_width=True
):

    if not description.strip():

        st.warning(
            "Please enter a ticket description."
        )

    else:

        try:

            # Get model prediction
            category, confidence = predict_ticket(
                description
            )

            # Generate suggested response
            response = generate_response(category)

            # Display prediction
            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Category: {category}"
            )

            # Display confidence
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(max(int(round(confidence)), 0), 100)
            )

            # Display customer response
            st.subheader(
                "💬 Suggested Customer Response"
            )

            st.info(response)

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# =====================================
# 7. FOOTER
# =====================================

st.divider()

st.caption(
    "Customer Support Ticket Classification System"
)

st.caption(
    "Developed using Python | Scikit-learn | Streamlit"
)
