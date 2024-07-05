import streamlit as st




st.set_page_config(
    page_title='NLP project')


st.sidebar.success('Выберите нужную страницу')

st.title('NLP project')
st.subheader('Классификация отзыва на фильм')
st.write(' 1. Классический ML-алгоритм, обученный на TF-IDF;') 
st.write('2. LSTM модель')
st.write('3. BERT-based')

st.subheader('Оценка степени токсичности сообщения с помощью модели rubert-tiny-toxicity')