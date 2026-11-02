
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="IncomeIQ",
    page_icon="💰",
    layout="centered"
)

# Load saved files
@st.cache_resource
def load_files():
    model = joblib.load("svm_income.pkl")
    scaler = joblib.load("income_scaler.pkl")
    columns = joblib.load("income_columns.pkl")

    return model, scaler, columns

model, scaler, columns = load_files()

num_cols = [
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

# Education mapping
education_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}

st.title("💰 IncomeIQ")
st.caption("AI-Powered Income Classification System")

st.write(
    "Enter your professional and financial details "
    "to predict the income category."
)

st.divider()

with st.form("income_form"):

    st.subheader("Personal & Professional Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=17,
            max_value=90,
            value=30
        )

        education = st.selectbox(
            "Education",
            list(education_map.keys()),
            index=12
        )

        workclass = st.selectbox(
            "Workclass",
            [
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked"
            ]
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Exec-managerial",
                "Prof-specialty",
                "Tech-support",
                "Sales",
                "Craft-repair",
                "Adm-clerical",
                "Other-service",
                "Machine-op-inspct",
                "Transport-moving",
                "Handlers-cleaners",
                "Farming-fishing",
                "Protective-serv",
                "Priv-house-serv",
                "Armed-Forces"
            ]
        )

    with col2:

        hours = st.slider(
            "Working Hours Per Week",
            1, 99, 40
        )

        capital_gain = st.number_input(
            "Capital Gain ($)",
            min_value=0,
            value=0,
            step=500
        )

        capital_loss = st.number_input(
            "Capital Loss ($)",
            min_value=0,
            value=0,
            step=100
        )

    submit = st.form_submit_button(
        "🔍 Predict Income",
        use_container_width=True
    )

if submit:

    # Construct all original 14 features
    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": 189778,
        "education": education,
        "educational-num": education_map[education],
        "marital-status": "Married-civ-spouse",
        "occupation": occupation,
        "relationship": "Husband",
        "race": "White",
        "gender": "Male",
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours,
        "native-country": "United-States"
    }])

    # One-hot encoding
    input_encoded = pd.get_dummies(input_data)

    # Match training columns and their order
    input_encoded = input_encoded.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale numerical features
    input_encoded[num_cols] = scaler.transform(
        input_encoded[num_cols]
    )

    # Prediction
    prediction = model.predict(input_encoded)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1 or prediction == ">50K":

        st.success("Predicted Income: Above $50K")

    else:

        st.info("Predicted Income: $50K or Below")

    st.caption(
        "This is an educational prediction based on "
        "the Adult Income dataset. Some unrequested "
        "features use representative default values."
    )
