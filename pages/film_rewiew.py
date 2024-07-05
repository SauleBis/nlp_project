import streamlit as st
import torch
import joblib
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from models.tf_edf_preprocessing import clean, lemmatize_text 
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

st.title('Классификация отзыва на фильм')
st.write('Введите текст для анализа')
uploaded_text = st.text_input('Введите текст')

label_dict = {1: 'Good', 0: 'Bad', 2: 'Neutral'}

if uploaded_text:
    # BERT + LogReg
    tokenizer_tiny2 = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model_tiny2 = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    # Токенизация введенного текста
    encoded_review = tokenizer_tiny2(uploaded_text, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    # st.write(encoded_rewies)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_tiny2 = model_tiny2.to(device)
    encoded_review = {key: value.to(device) for key, value in encoded_review.items()}
    # Получение векторного представления отзыва
    with torch.no_grad():
        model_out = model_tiny2(**encoded_review)
        vector = model_out.last_hidden_state[:, 0, :].cpu().numpy()
    # Загрузка LogReg 
    bert_log_reg = joblib.load('models/bert_log_reg.pkl')
    start_time_blr = time.time()
    predict_log_reg = bert_log_reg.predict(vector)
    end_time_blr = time.time()
    predict_label = label_dict[predict_log_reg[0]]
    st.write(f'Токенизация BERT и LogReg, предсказанная категория отзыва - {predict_label}')
    st.write(f'Время предсказания {round((end_time_blr - start_time_blr), 4)}')
   
    # TF-EDF + LogReg 
 #   stop_words_tf = stopwords.words('russian')
 #   cleaned_text = clean(uploaded_text)
  #  lem_text = lemmatize_text(cleaned_text)
    # st.write(lem_text)
  # reg_tokenizer = RegexpTokenizer('\w+')
  #  tokenized_text = reg_tokenizer.tokenize_sents(lem_text)
   # st.write(tokenized_text)
  #  tokenized_text = ' '.join(tokenized_text)
  #  vectorizer = TfidfVectorizer(
  #  max_df=2,
  #  min_df=1,
  #  stop_words=stop_words_tf
  #  t0 = time.time()
  #  X_tfidf = vectorizer.fit_transform([lem_text]) 
    #st.write(X_tfidf)
  #  model_tfidf = joblib.load('models/logreg_model.pkl')
  #  start_time_tf = time.time()
  #  predict_tf = model_tfidf.predict(X_tfidf)
  #  end_time_tf = time.time()
  #  predict_label_tf = label_dict[predict_tf[0]]
  #  st.write(f'Tf-Idf и LogReg, предсказанная категория отзыва - {predict_label_tf}')
  #  st.write(f'Время предсказания {round((end_time_tf - start_time_tf), 4)}')

data_results = {
    'Model': ['LogReg + Tf-IDF', 'LogReg + BERT', 'LSTM'],
    'Accuracy': [0.82, 0.76, 0.63]}
df = pd.DataFrame(data_results)
st.title('Сравнение моделей')
st.table(df)




