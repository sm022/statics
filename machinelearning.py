import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt  # matplotlib를 사용하여 그래프 생성

# CSV 파일에서 데이터를 로드합니다.
df = pd.read_csv('C:/Users/smyoo/PycharmProjects/pythonProject/reviews.csv')

# 'Sentiment' 열을 기준으로 긍정과 부정으로 데이터를 분리합니다.
positive_data = df[df['Sentiment'] == 1]  # 'Sentiment'가 1인 데이터는 긍정 리뷰입니다.
negative_data = df[df['Sentiment'] == 0]  # 'Sentiment'가 0인 데이터는 부정 리뷰입니다.

# 긍정 리뷰와 부정 리뷰를 따로 저장할 CSV 파일의 경로를 설정합니다.
positive_data.to_csv('positive_reviews.csv', index=True)  # index=True로 설정하여 인덱스를 함께 저장합니다.
negative_data.to_csv('negative_reviews.csv', index=True)

# 각 클래스에 해당하는 데이터를 추출합니다.
positive_texts = positive_data['Review']
negative_texts = negative_data['Review']
positive_labels = positive_data['Sentiment']
negative_labels = negative_data['Sentiment']

# 긍정 리뷰와 부정 리뷰 데이터를 훈련 및 테스트 세트로 분할합니다.
X_train_positive, X_test_positive, y_train_positive, y_test_positive = train_test_split(positive_texts, positive_labels, test_size=0.2, random_state=42)
X_train_negative, X_test_negative, y_train_negative, y_test_negative = train_test_split(negative_texts, negative_labels, test_size=0.2, random_state=42)

# 훈련 데이터와 테스트 데이터를 병합합니다.
X_train = pd.concat([X_train_positive, X_train_negative])
X_test = pd.concat([X_test_positive, X_test_negative])
y_train = pd.concat([y_train_positive, y_train_negative])
y_test = pd.concat([y_test_positive, y_test_negative])

# TF-IDF 벡터화를 수행합니다.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # 최대 특성 수를 설정합니다.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 로지스틱 회귀 모델을 생성하고 훈련합니다.
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# 테스트 데이터에 대한 예측을 수행합니다.
y_pred = lr_model.predict(X_test_tfidf)

# 모델의 성능을 평가합니다.
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')

# 모델로부터 예측된 결과
predicted_sentiment = ["Positive" if y == 1 else "Negative" for y in y_pred]

# 그래프 생성
plt.figure(figsize=(8, 6))
plt.scatter(accuracy_score(y_test, y_pred), classification_rep, marker='o', s=100)
plt.xlabel('Accuracy')
plt.ylabel('Classification Report')
plt.title('Accuracy vs Classification Report')
plt.grid(True)
plt.tight_layout()

# 이미지를 저장합니다.
plt.savefig("C:/Users/smyoo/PycharmProjects/pythonProject/result_image.png")

# 이미지를 표시하려면 아래 코드 사용
# plt.show()
