import pandas as pd

# CSV 파일에서 데이터를 로드합니다. 데이터 파일의 경로를 적절히 수정하세요.
df = pd.read_csv('C:/Users/smyoo/PycharmProjects/pythonProject/reviews.csv')

# 'Sentiment' 열을 기준으로 긍정과 부정으로 데이터를 분리하기 전에 Sentiment 값을 이진 분류로 변환합니다.
df['Sentiment'] = df['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

# 'Sentiment' 열을 기준으로 긍정과 부정으로 데이터를 분리합니다.
positive_data = df[df['Sentiment'] == 1]  # 이제 'Sentiment'가 1인 데이터는 긍정 리뷰입니다.
negative_data = df[df['Sentiment'] == 0]  # 이제 'Sentiment'가 0인 데이터는 부정 리뷰입니다.

# 긍정 리뷰와 부정 리뷰를 따로 저장할 CSV 파일의 경로를 설정합니다.
positive_data.to_csv('positive_reviews.csv', index=True)  # index=True로 설정하여 인덱스를 함께 저장합니다.
negative_data.to_csv('negative_reviews.csv', index=True)

print("긍정 리뷰와 부정 리뷰를 분리하여 저장했습니다.")
