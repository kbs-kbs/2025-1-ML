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
- **StandardScaler**(): 특성 표준화. 데이터를 표준편차가 1인 정규분포를 따르는 수들로 변환합니다. 표준편차란 평균에 대한 오차값입니다.
- **LinearSVC**():

RBF : 등고선을 활용한 분리

true, 나머지 false 형태로 데이터 분리할 때 좋음

라벨간 중요도가 동일할 경우에는 선형분류가 나을 수도 있음


> [!Note]
> LinearSVC를 통해 모델을 학습시키는 것을 파인튜닝이라 할 수 있는가?
> No. 이 코드에서 사용된 LinearSVC는 pre-trained 모델이 아니며, 스타터 모델(즉, 초기 상태의 모델)로 간주됩니다. 따라서 이 코드는 **파인튜닝(fine-tuning)**이라고 할 수 없습니다. 이유를 아래에 자세히 설명하겠습니다.
> 
> 1. 파인튜닝(fine-tuning)의 정의
> 파인튜닝은 이미 학습된(pre-trained) 모델을 기반으로 특정 데이터셋에 맞게 추가 학습을 수행하는 과정을 의미합니다. 일반적으로 대규모 데이터셋에서 학습된 모델을 가져와, 새로운 데이터셋에 맞춰 일부 레이어를 고정하거나 전체를 재학습하는 방식으로 진행됩니다. 딥러닝에서 주로 사용되는 용어이며, 예를 들어:
>
> 사전 학습된 이미지 분류 모델(예: ResNet, VGG)을 가져와 특정 데이터셋에 맞게 재학습.
>
> 언어 모델(예: BERT, GPT)을 특정 텍스트 데이터에 맞게 조정.
> 
> 2. LinearSVC는 스타터 모델
> LinearSVC는 사이킷런에서 제공하는 선형 서포트 벡터 머신(SVM) 모델로, 학습 전에 아무런 사전 학습된 가중치나 파라미터를 가지고 있지 않습니다. 즉, 이 모델은 **처음부터 시작(starter)**하여 제공된 데이터셋(X, y)을 기반으로 학습을 수행합니다.
>
> 따라서 이 코드에서 LinearSVC는:
> 사전 학습된(pre-trained) 상태가 아니며, 주어진 데이터셋으로 처음부터 학습을 시작합니다. 이 과정은 일반적인 **모델 훈련(training)**이지, 파인튜닝이라고 볼 수 없습니다.
>
> 3. 결론
> 이 코드에서 LinearSVC를 통해 학습하는 과정은 파인튜닝이 아니라 **모델 초기화 후 학습(training)**입니다. 파인튜닝은 사전 학습된(pre-trained) 모델을 사용하는 경우에만 해당됩니다.
> 
> 만약 파인튜닝을 수행하려면, 이미 다른 데이터셋에서 학습된 가중치를 가진 모델을 불러와 이를 새로운 데이터셋에 맞게 조정해야 합니다.
>
