import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Encoding maps (must match training) ─────────────────────────────────────
FREQ_FLYER_MAP   = {"No": 0, "Yes": 1}
INCOME_MAP       = {"High Income": 0, "Low Income": 1, "Middle Income": 2, "No Income": 3}
SOCIAL_MAP       = {"No": 0, "Yes": 1}
HOTEL_MAP        = {"No": 0, "Yes": 1}

# ── Sidebar – Input Form ─────────────────────────────────────────────────────
st.sidebar.header("🛫 Customer Profile")
st.sidebar.markdown("Fill in the customer details below:")

age = st.sidebar.slider("Age", min_value=27, max_value=38, value=31, step=1)

frequent_flyer = st.sidebar.selectbox(
    "Frequent Flyer?",
    options=["No", "Yes"],
    help="Has the customer opted into the frequent flyer programme?"
)

annual_income = st.sidebar.selectbox(
    "Annual Income Class",
    options=["High Income", "Low Income", "Middle Income", "No Income"]
)

services_opted = st.sidebar.slider(
    "Number of Services Opted",
    min_value=1, max_value=6, value=3, step=1,
    help="Total travel services subscribed (flights, hotels, insurance, etc.)"
)

social_media = st.sidebar.selectbox(
    "Account Synced to Social Media?",
    options=["No", "Yes"]
)

booked_hotel = st.sidebar.selectbox(
    "Booked Hotel?",
    options=["No", "Yes"]
)

predict_btn = st.sidebar.button("🔮 Predict Churn", use_container_width=True, type="primary")

# ── Main content ─────────────────────────────────────────────────────────────
st.title("✈️ Customer Churn Prediction")
st.markdown("**Random Forest Classifier | B.Tech Gen AI – Final Project**")
st.divider()

# Overview section
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Random Forest")
col2.metric("Training Accuracy", "~87%")
col3.metric("AUC-ROC", "~0.90")

st.divider()

# ── Prediction ───────────────────────────────────────────────────────────────
if predict_btn:
    # Build input DataFrame
    input_data = pd.DataFrame([{
        "Age":                        age,
        "FrequentFlyer":              FREQ_FLYER_MAP[frequent_flyer],
        "AnnualIncomeClass":          INCOME_MAP[annual_income],
        "ServicesOpted":              services_opted,
        "AccountSyncedToSocialMedia": SOCIAL_MAP[social_media],
        "BookedHotelOrNot":           HOTEL_MAP[booked_hotel],
    }])

    prediction   = model.predict(input_data)[0]
    probability  = model.predict_proba(input_data)[0]
    churn_prob   = probability[1] * 100
    no_churn_prob = probability[0] * 100

    st.subheader("🎯 Prediction Result")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **HIGH CHURN RISK**")
            st.markdown(f"This customer is **likely to churn** with a confidence of **{churn_prob:.1f}%**.")
        else:
            st.success("✅ **LOW CHURN RISK**")
            st.markdown(f"This customer is **likely to stay** with a confidence of **{no_churn_prob:.1f}%**.")

        # Probability bar
        st.markdown("**Churn Probability**")
        st.progress(int(churn_prob))
        st.caption(f"Churn: {churn_prob:.1f}%  |  No Churn: {no_churn_prob:.1f}%")

    with res_col2:
        # Mini pie chart
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.pie(
            [no_churn_prob, churn_prob],
            labels=["No Churn", "Churn"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11}
        )
        ax.set_title("Churn Probability", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Input summary
    st.subheader("📋 Customer Input Summary")
    summary = pd.DataFrame({
        "Feature": ["Age", "Frequent Flyer", "Annual Income Class",
                    "Services Opted", "Account Synced to Social Media", "Booked Hotel"],
        "Value": [age, frequent_flyer, annual_income,
                  services_opted, social_media, booked_hotel]
    })
    st.table(summary)

    # Recommendation
    st.subheader("💡 Business Recommendation")
    if prediction == 1:
        st.warning("""
**Recommended Actions for At-Risk Customer:**
- 🎁 Offer a **personalized loyalty discount** or upgrade.
- 📞 Schedule a **proactive retention call**.
- 🏨 Promote **hotel bundling deals** to increase engagement.
- ✈️ Enrol in the **Frequent Flyer Programme** if not already a member.
- 📱 Encourage **social media account sync** for better personalised offers.
""")
    else:
        st.info("""
**Customer Appears Healthy – Continue Engagement:**
- ⭐ Reward with **loyalty points** to reinforce positive behaviour.
- 📊 Monitor service usage trends quarterly.
- 💌 Send **personalised travel offers** to maintain satisfaction.
""")

else:
    # Default welcome state
    st.markdown("""
    ### 👋 Welcome to the Customer Churn Predictor

    This application uses a **Random Forest machine learning model** trained on travel customer data 
    to predict whether a customer is at risk of churning.

    **How to use:**
    1. Fill in the customer details in the **sidebar** on the left.
    2. Click the **"Predict Churn"** button.
    3. View the prediction result, probability scores, and business recommendations.

    ---
    """)

    # Feature descriptions
    st.subheader("📊 Dataset Features")
    feat_df = pd.DataFrame({
        "Feature": ["Age", "FrequentFlyer", "AnnualIncomeClass",
                    "ServicesOpted", "AccountSyncedToSocialMedia", "BookedHotelOrNot"],
        "Type": ["Numeric", "Categorical (Yes/No)", "Categorical (4 classes)",
                 "Numeric (1–6)", "Categorical (Yes/No)", "Categorical (Yes/No)"],
        "Description": [
            "Customer age (27–38)",
            "Whether the customer is a frequent flyer",
            "Annual income bracket of the customer",
            "Number of travel services opted by customer",
            "Whether account is synced to social media",
            "Whether the customer has booked a hotel"
        ]
    })
    st.table(feat_df)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ using Streamlit | B.Tech Gen AI – Customer Churn Prediction Project")
