import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Premium CSS
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.main {
    background-color: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 15px;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Student Performance Dashboard")
st.caption("Machine Learning Based Prediction System")

st.markdown("---")

# Sidebar
st.sidebar.header("Input Parameters")

study_hours = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0)
attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 75.0)
prev_score = st.sidebar.slider("Previous Score", 0.0, 100.0, 60.0)
sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
extra_classes = st.sidebar.slider("Extra Classes", 0, 10, 2)

# Tabs
tab1, tab2 = st.tabs(["Single Prediction", "Compare Students"])

# =========================
# SINGLE PREDICTION
# =========================
with tab1:

    if st.button("Predict"):

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

        st.subheader(f"Predicted Marks: {prediction:.2f}")

        if prediction > 80:
            st.success("Excellent performance expected")
        elif prediction > 60:
            st.warning("Average performance")
        else:
            st.error("Needs improvement")

        # Suggestions
        st.markdown("### Suggestions")
        if study_hours < 4:
            st.warning("Increase study hours")
        if sleep_hours < 6:
            st.warning("Improve sleep schedule")
        if attendance < 70:
            st.warning("Maintain better attendance")

        # Plotly Chart (interactive)
        labels = ['Study', 'Attendance', 'Previous', 'Sleep', 'Extra']
        values = [study_hours, attendance/10, prev_score/10, sleep_hours*2, extra_classes*2]

        df_plot = pd.DataFrame({
            "Feature": labels,
            "Value": values
        })

        fig = px.bar(df_plot, x="Feature", y="Value", title="Input Contribution")
        st.plotly_chart(fig, use_container_width=True)


# =========================
# COMPARISON
# =========================
with tab2:

    st.subheader("Compare performance of two students")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Student A")
        a_study = st.slider("Study Hours A", 0.0, 12.0, 5.0)
        a_att = st.slider("Attendance A", 0.0, 100.0, 75.0)
        a_prev = st.slider("Previous Score A", 0.0, 100.0, 60.0)
        a_sleep = st.slider("Sleep Hours A", 0.0, 12.0, 7.0)
        a_extra = st.slider("Extra Classes A", 0, 10, 2)

    with col2:
        st.markdown("Student B")
        b_study = st.slider("Study Hours B", 0.0, 12.0, 6.0)
        b_att = st.slider("Attendance B", 0.0, 100.0, 80.0)
        b_prev = st.slider("Previous Score B", 0.0, 100.0, 70.0)
        b_sleep = st.slider("Sleep Hours B", 0.0, 12.0, 6.0)
        b_extra = st.slider("Extra Classes B", 0, 10, 3)

    if st.button("Compare"):

        def predict(s, a, p, sl, e):
            eff = s * a
            health = sl * 10

            df = pd.DataFrame([[s, a, p, sl, e, eff, health]],
                columns=[
                    'Study_Hours', 'Attendance_%', 'Previous_Score',
                    'Sleep_Hours', 'Extra_Classes',
                    'Study_Efficiency', 'Health_Index'
                ])

            scaled = scaler.transform(df)
            pred = model.predict(scaled)
            return np.clip(pred, 0, 100)[0]

        pred_a = predict(a_study, a_att, a_prev, a_sleep, a_extra)
        pred_b = predict(b_study, b_att, b_prev, b_sleep, b_extra)

        st.markdown(f"Student A: {pred_a:.2f}")
        st.markdown(f"Student B: {pred_b:.2f}")

        if pred_a > pred_b:
            st.success("Student A is performing better")
        else:
            st.success("Student B is performing better")

        # Plotly comparison
        df_compare = pd.DataFrame({
            "Student": ["A", "B"],
            "Marks": [pred_a, pred_b]
        })

        fig2 = px.bar(df_compare, x="Student", y="Marks", title="Comparison Result")
        st.plotly_chart(fig2, use_container_width=True)
