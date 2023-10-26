import pandas as pd
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords

# nltk의 리소스를 다운로드합니다.
nltk.download('punkt')
nltk.download('stopwords')

# CSV 파일에서 데이터를 로드합니다.
df = pd.read_csv('C:/Users/smyoo/PycharmProjects/pythonProject/reviews.csv')

# 리뷰 텍스트가 있는 열의 이름을 'review'로 가정합니다.
texts = df['Review'].dropna()  # 결측값을 제거합니다.

# 모든 리뷰를 하나의 텍스트로 합칩니다.
all_texts = ' '.join(texts)

# 텍스트를 소문자로 변환하고 토큰화합니다.
tokens = nltk.word_tokenize(all_texts.lower())

# 불용어를 제거합니다.
tokens = [word for word in tokens if word not in stopwords.words('english')]

# 1. 단어 빈도 분석
word_freq = FreqDist(tokens)

# 2. 연관 단어 분석
bigram_finder = BigramCollocationFinder.from_words(tokens)
bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)

# 3. n-gram 분석
n = 3
n_grams = list(ngrams(tokens, n))
ngram_freq = FreqDist(n_grams)

# 결과를 저장할 파일의 경로를 설정합니다.
file_path = "C:/Users/smyoo/PycharmProjects/pythonProject/analysis_results.txt"

# 파일을 쓰기 모드로 엽니다.
with open(file_path, 'w', encoding='utf-8') as file:
    # 1. 단어 빈도 분석 결과를 파일에 작성합니다.
    file.write("Word Frequency:\n")
    for word, freq in word_freq.most_common(10):
        file.write(f"{word}: {freq}\n")

    # 2. 연관 단어 분석 결과를 파일에 작성합니다.
    file.write("\nBigrams:\n")
    for bigram in bigrams:
        file.write(f"{' '.join(bigram)}\n")

    # 3. n-gram 분석 결과를 파일에 작성합니다.
    file.write(f"\n{n}-grams:\n")
    for ngram, freq in ngram_freq.most_common(10):
        file.write(f"{' '.join(ngram)}: {freq}\n")

# 파일이 성공적으로 저장되었는지 확인하기 위해 메시지를 출력합니다.
print(f"Analysis results have been saved to {file_path}")
