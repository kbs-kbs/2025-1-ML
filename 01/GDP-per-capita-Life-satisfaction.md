## 목표
진찰 기록으로 폐암 수술 환자의 생존율을 예측하는 모델 생성



## 사용 라이브러리
|언어|버전|라이브러리|버전|사용 모듈|용도|
|---|---|---|---|---|---|
|Python|3.11.11|tensorflow.keras.models|2.18.0|Sequential|전체 모델 구조를 정의|
|||tensorflow.keras.layers|2.18.0|Dense|모델 내부의 개별 층을 구성|
|||numpy|1.26.4||데이터 불러오기|



## 외부 데이터 소스 의존성
- TensorFlow: 오프라인 환경에서 GPU 가속을 사용하는 경우, CUDA와 cuDNN 라이브러리가 필요합니다. 이들은 NVIDIA 웹사이트에서 다운로드해야 하며, 설치를 위해 미리 준비해야 합니다.

- NumPy: 고성능 연산을 위해 BLAS/LAPACK 라이브러리(OpenBLAS, MKL 등)에 의존합니다. 대부분의 배포판에 포함되어 있지만, 오프라인 환경에서는 별도로 설치해야 할 수도 있습니다.

- 수술 환자 데이터: `!git clone https://github.com/taehojo/data.git`

## 단계

- model.compile(): 모델 학습 준비를 설정한다.
- model.fit(): 모델을 학습시킨다.
- model.predict(): 모델로 예측(추론)을 수행한다.


```python
# prompt: 맷플롯립, 판다스, 사이킷런을 이용한 1인당 GDP라는 특성 하나를 가진 삶의 만족도에 대한 선형 모델

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# 데이터 로드 (예시 데이터)
lifesat = pd.read_csv("https://raw.githubusercontent.com/ageron/data/refs/heads/main/lifesat/lifesat.csv")

# 특성(X)과 타겟(y) 설정
X = lifesat[['GDP per capita (USD)']].values
y = lifesat['Life satisfaction'].values

# 데이터 시각화
lifesat.plot(kind='scatter', grid=True, x='GDP per capita (USD)', y='Life satisfaction')
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# OECD 데이터에 없는 키프로스(Cyprus) 사람들이 얼마나 행복한지 알아보기 위해 이 모델을 사용
X_new = [[37_655.2]]
print(model.predict(X_new))
```
