import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #e2e8f0;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 2.8em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Student Performance Prediction Dashboard")
st.write("Model Accuracy (R² Score): 0.83")

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["Single Prediction", "Compare Students"])

# =========================
# TAB 1: Single Prediction
# =========================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study Hours", 0.0, 12.0, 5.0)
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
        prev_score = st.slider("Previous Score", 0.0, 100.0, 60.0)

    with col2:
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
        extra_classes = st.slider("Extra Classes", 0, 10, 2)

    if st.button("Predict", key="single"):

        study_efficiency = study_hours * attendance
        health_index = sleep_hours * 10

        student_data = pd.DataFrame(
            [[study_hours, attendance, prev_score, sleep_hours, extra_classes,
              study_efficiency, health_index]],
            columns=[
                'Study_Hours', 'Attendance_%', 'Previous_Score',
                'Sleep_Hours', 'Extra_Classes',
                'Study_Efficiency', 'Health_Index'
            ]
        )

        scaled = scaler.transform(student_data)
        prediction = model.predict(scaled)
        prediction = np.clip(prediction, 0, 100)[0]

        st.markdown(f"### Predicted Marks: {prediction:.2f}")

        # Performance message
        if prediction > 80:
            st.success("Excellent performance expected")
        elif prediction > 60:
            st.warning("Average performance")
        else:
            st.error("Needs improvement")

        # Graph
        labels = ['Study', 'Attendance', 'Previous', 'Sleep', 'Extra']
        values = [study_hours, attendance/10, prev_score/10, sleep_hours*2, extra_classes*2]

        fig, ax = plt.subplots()
        ax.plot(labels, values, marker='o')
        ax.set_title("Input Contribution Overview")
        st.pyplot(fig)

# =========================
# TAB 2: Compare Students
# =========================
with tab2:

    st.write("Compare performance of two students")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student A")
        a_study = st.slider("Study Hours A", 0.0, 12.0, 5.0)
        a_att = st.slider("Attendance A", 0.0, 100.0, 75.0)
        a_prev = st.slider("Previous Score A", 0.0, 100.0, 60.0)
        a_sleep = st.slider("Sleep Hours A", 0.0, 12.0, 7.0)
        a_extra = st.slider("Extra Classes A", 0, 10, 2)

    with col2:
        st.subheader("Student B")
        b_study = st.slider("Study Hours B", 0.0, 12.0, 6.0)
        b_att = st.slider("Attendance B", 0.0, 100.0, 80.0)
        b_prev = st.slider("Previous Score B", 0.0, 100.0, 70.0)
        b_sleep = st.slider("Sleep Hours B", 0.0, 12.0, 6.0)
        b_extra = st.slider("Extra Classes B", 0, 10, 3)

    if st.button("Compare"):

        def predict(s, a, p, sl, e):
            eff = s * a
            health = sl * 10

            df = pd.DataFrame(
                [[s, a, p, sl, e, eff, health]],
                columns=[
                    'Study_Hours', 'Attendance_%', 'Previous_Score',
                    'Sleep_Hours', 'Extra_Classes',
                    'Study_Efficiency', 'Health_Index'
                ]
            )

            scaled = scaler.transform(df)
            pred = model.predict(scaled)
            return np.clip(pred, 0, 100)[0]

        pred_a = predict(a_study, a_att, a_prev, a_sleep, a_extra)
        pred_b = predict(b_study, b_att, b_prev, b_sleep, b_extra)

        st.markdown(f"### Student A: {pred_a:.2f}")
        st.markdown(f"### Student B: {pred_b:.2f}")

        fig, ax = plt.subplots()
        ax.bar(['Student A', 'Student B'], [pred_a, pred_b])
        ax.set_title("Predicted Marks Comparison")
        st.pyplot(fig)
