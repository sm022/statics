from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from textblob import TextBlob
from gensim import corpora, models
import string
import csv
import time


def get_reviews(url):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(5)  # 페이지 로딩을 기다립니다.

    try:
        # JavaScript를 사용하여 페이지를 스크롤합니다.
        for _ in range(30):  # 스크롤 횟수는 조절이 필요할 수 있습니다.
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # 로드를 기다립니다.

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        review_elements = soup.find_all('div', class_='apphub_UserReviewCardContent')
        reviews = [element.find('div', class_='apphub_CardTextContent').get_text(strip=True) for element in review_elements]

    finally:
        driver.quit()

    return reviews


def translate_review(text):
    translator = Translator()
    return translator.translate(text, dest='en').text

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def apply_topic_modeling(reviews, num_topics=5):
    processed_reviews = [preprocess_text(review).split() for review in reviews]
    dictionary = corpora.Dictionary(processed_reviews)
    corpus = [dictionary.doc2bow(review) for review in processed_reviews]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

# 함수를 호출하여 리뷰를 가져옵니다.
url = "https://steamcommunity.com/app/1868140/reviews/?browsefilter=toprated&snr=1_5_100010_"
reviews = get_reviews(url)

# 리뷰의 감정을 분석합니다.
sentiments = [analyze_sentiment(review) for review in reviews]

# 리뷰에 토픽 모델링을 적용합니다.
apply_topic_modeling(reviews)

# 리뷰와 감정 분석 결과를 CSV 파일로 저장합니다.
with open("C:/Users/smyoo/PycharmProjects/pythonProject/reviews.csv", mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Review", "Sentiment"])
    for review, sentiment in zip(reviews, sentiments):
        writer.writerow([review, sentiment])
