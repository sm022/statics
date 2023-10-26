import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# CSV 파일에서 데이터 읽기
def read_csv(file_path):
    return pd.read_csv(file_path)

# WordCloud 생성 및 파일로 저장
def generate_and_save_wordcloud(text, file_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file(file_path)

# CSV 파일 경로
csv_file_path = "C:/Users/smyoo/PycharmProjects/pythonProject/reviews.csv"

# CSV 파일 읽기
data = read_csv(csv_file_path)

# 긍정 리뷰와 부정 리뷰 분리
positive_reviews = data[data['Sentiment'] > 0]['Review']
negative_reviews = data[data['Sentiment'] < 0]['Review']

# 긍정 리뷰 WordCloud 생성 및 저장
generate_and_save_wordcloud(' '.join(positive_reviews), "C:/Users/smyoo/PycharmProjects/pythonProject/positive_wordcloud.png")

# 부정 리뷰 WordCloud 생성 및 저장
generate_and_save_wordcloud(' '.join(negative_reviews), "C:/Users/smyoo/PycharmProjects/pythonProject/negative_wordcloud.png")
