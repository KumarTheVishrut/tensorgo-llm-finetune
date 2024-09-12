import streamlit as st

st.title("LLM Evaluation tensorgo")

st.text_area("Enter text for evaluation:", "Type here...")

st.subheader("Base Model Output")
st.write("This is where the base model output will be displayed.")

st.subheader("Fine-Tuned Model Output")
st.write("This is where the fine-tuned model output will be displayed.")

st.button("Evaluate")
