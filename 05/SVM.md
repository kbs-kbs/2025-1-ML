서포트 벡터 머신: 선을 그어 분리    

하드 마진 분류: 이상치 하나도 없게   
소프트 마진 분류: 이상치 허용     

하이퍼 파라미터 C: 마진 오류와 결정 경계의 간격을 조절하는 역할
C가 클수록 중립지대가 줄어듦


```python
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # Iris virginica

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, dual=True, random_state=42))
svm_clf.fit(X, y)
```

데이터 스케일링의 두 가지 주요 방법 정규화(Normalization)와 표준화(Standardization) 중 표준화를 사용하고 있음
스케일링은 모델의 가정 충족과 계산 성능의 향상을 위해 필요합니다.

- **make_pipeline**(*step): 가변인자로 여러 단계를 받는다. 주로 전처리 함수와 학습 함수 두개를 넘긴다.


StandardScaler(): 특성 표준화. 데이터를 표준편차가 1인 정규분포를 따르는 수들로 변환합니다. 표준편차란 평균에 대한 오차값입니다.


RBF : 등고선을 활용한 분리

true, 나머지 false 형태로 데이터 분리할 때 좋음

라벨간 중요도가 동일할 경우에는 선형분류가 나을 수도 있음
