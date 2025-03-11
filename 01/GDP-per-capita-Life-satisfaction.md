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
