import torch
import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st



model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba



st.title('Оценка степени токсичности сообщения с помощью модели rubert-tiny-toxicity')
st.write('Введите текст для анализа')
uploaded_text = st.text_input('Введите текст')
classify = st.button('Analize')

if uploaded_text and classify:
     toxic_score = text2toxicity(uploaded_text)
     st.write(f'Вероятность токсичности сообщения {toxic_score}')