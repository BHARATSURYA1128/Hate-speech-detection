import streamlit as st
import requests

st.set_page_config(page_title="ContextGuard", layout="centered")

st.title("🛡️ ContextGuard")
st.subheader("AI Hate Speech Detection")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        try:
            response = requests.post(
                "https://your-app.onrender.com/predict",
                json={"text": text}
            )

            result = response.json()

            if "prediction" in result:
                st.success(f"Prediction: {result['prediction']}")
            else:
                st.error("Error: " + result.get("error", "Unknown"))

        except:
            st.error("API not reachable")