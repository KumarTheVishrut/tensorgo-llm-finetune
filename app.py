import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=False, 
    torch_dtype=torch.float32,  
)
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

ft_model = PeftModel.from_pretrained(base_model, "/home/vishrut/projs/tensorgo-proj/phi2-tensorgo-challenge")

st.title("LLM Evaluation for tensor-Go")

user_input = st.text_area("Enter text for evaluation:", "Type here...")

if st.button("Evaluate"):
    if user_input:
        model_input = eval_tokenizer(user_input, return_tensors="pt")

        base_model.eval()
        with torch.no_grad():
            base_output = eval_tokenizer.decode(
                base_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.11)[0],
                skip_special_tokens=True
            )

        ft_model.eval()
        with torch.no_grad():
            ft_output = eval_tokenizer.decode(
                ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.11)[0],
                skip_special_tokens=True
            )

        st.subheader("Base Model Output")
        st.write(base_output)
        st.subheader("Fine-Tuned Model Output")
        st.write(ft_output)
    else:
        st.error("Please enter some text for evaluation.")
