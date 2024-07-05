import re
import time
import string
from tqdm import tqdm
import pymorphy3
from pymorphy3 import MorphAnalyzer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
nltk.download('punkt')


def clean(text):
    text = re.sub(r'([А-Я])', r' \1', text) # делаем пробел между верхним и нижним регистром(123/473)
    text = text.lower() # нижний регистр (194)
    text = re.sub(r'[\w\.-]+\.ru', " ", text) # удаляем сайты (4)
    text = re.sub(r'http\S+', " ", text) # удаляем ссылки (10/1076)
    text = re.sub(r'@\w+',' ',text) # удаляем упоминания пользователей (1937/1923)
    text = re.sub(r'#\w+', ' ', text) # удаляем хэштеги (10/77/1923)
    text = re.sub(r'\d+', ' ', text) # удаляем числа()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Удаление английских слов
    text = ' '.join(re.findall(r'\b[а-яА-ЯёЁ]+\b', text))
    # Удаление стоп-слов
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word.lower() not in stop_words])
    # text = re.sub(r'<.*?>',' ', text) #
    text = re.sub(r'['u'\U0001F600-\U0001F64F'
                  u'\U0001F300-\U0001F5FF'
                  u'\U0001F680-\U0001F6FF'
                  u'\U0001F1E0-\U0001F1FF'
                  u'\U00002500-\U00002BEF'
                  u'\U00002702-\U000027B0'
                  u'\U000024C2-\U0001F251'
                  u'\U0001f926-\U0001f937'
                  u'\U00010000-\U0010ffff'
                    ']+', '', text, flags=re.UNICODE) # удаляем эмоджи(4/1707)
    return text

morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    """Лемматизация текста."""
    tokens = word_tokenize(text)
    lemmatized_text = ' '.join([morph.parse(word)[0].normal_form for word in tokens])
    return lemmatized_text