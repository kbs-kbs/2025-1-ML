# 비선형 SVM (커널 · 유사도 특성)

> SVM 2부 — 선형 SVM(`05/SVM.md`)에 이어, 비선형 데이터를 다루는 방법

## 1. 선형 SVM 복습

- **SVM(서포트 벡터 머신)**: 매우 강력하고 선형·비선형 분류, 회귀, 이상치 탐색에도 사용할 수 있는 다목적 머신러닝 모델
- **라지 마진 분류(large margin classification)**: 두 클래스 사이의 도로(마진)를 최대한 넓게
- **서포트 벡터(support vector)**: 도로의 경계를 결정하는, 경계에 가장 가까운 샘플
- 특성 스케일에 민감 → `StandardScaler` 와 파이프라인으로 묶어 사용

| 구분 | 내용 |
|---|---|
| 하드 마진 | 이상치를 하나도 허용하지 않음. 선형적으로 구분 가능해야 하고 이상치에 민감 |
| 소프트 마진 | 도로 폭을 넓게 유지하는 것과 **마진 오류(margin violation)** 사이의 균형. `C` 로 조절 |

- `C` 를 줄이면 도로가 커지고 마진 오류가 많아짐 → 서포트 벡터가 많아져 **과대적합 위험 감소**
- 너무 많이 줄이면 **과소적합**. SVM이 과대적합이면 C를 감소시켜 규제

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, dual=True, random_state=42))
svm_clf.fit(X, y)
```

- `decision_function()`: 각 샘플과 결정 경계 사이의 거리(양수/음수)를 반환

## 2. 비선형 데이터 다루기 — 다항 특성

- 비선형 데이터는 **다항 특성과 같은 특성을 더 추가**하면 선형적으로 구분되는 데이터셋이 될 수 있음
- moons 데이터셋에 `PolynomialFeatures` + `StandardScaler` + `LinearSVC` 파이프라인 적용

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, dual=True, random_state=42)
)
polynomial_svm_clf.fit(X, y)
```

- 3차 다항 특성을 추가하면 선형 SVM으로도 moons 데이터를 분리할 수 있음

## 3. 다항식 커널 (kernel trick)

- 특성을 실제로 추가하지 않고, **매우 높은 차수의 다항 특성을 많이 추가한 것과 같은 결과**를 얻는 기법
- 계산량이 훨씬 적음

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_kernel_svm_clf.fit(X, y)
```

| 하이퍼파라미터 | 의미 |
|---|---|
| `kernel="poly"` | 다항식 커널 |
| `degree` | 다항식 차수. 높을수록 변곡점이 많아져 더 복잡한 경계 |
| `coef0` | 커널 함수의 상수항(높은 차수 항의 영향 조절) |
| `C` | 규제. 크면 과대적합, 작으면 과소적합 |

## 4. 유사도 특성 (RBF)

- **유사도 함수(similarity function)** 로 계산한 특성을 추가 → **랜드마크(landmark)** 와 얼마나 닮았는지 측정
- 1차원 데이터셋에 랜드마크 x₁ = -2, x₁ = 1을 추가하고 **γ = 0.3인 가우스 방사 기저 함수(RBF)** 를 유사도 함수로 정의

$$ \phi_{\gamma}(x, \ell) = \exp(-\gamma \lVert x - \ell \rVert^2) $$

- 랜드마크에서 멀어질수록 0에 가까워지고, 가까우면 1에 가까워짐 → 선형 분리가 가능해짐
- γ(gamma)는 **영향 범위**를 결정
  - γ를 크게 → 각 샘플의 영향 범위가 좁아져 경계가 구불구불(과대적합 위험)
  - γ를 작게 → 영향 범위가 넓어져 경계가 완만(과소적합 위험)

```python
rbf_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=5, C=0.001)
)
rbf_kernel_svm_clf.fit(X, y)
```

## 5. 어떤 커널을 쓸까?

1. 먼저 **선형 SVM(`LinearSVC`)** 을 시도 (빠르고 데이터가 많을 때 유리)
2. 잘 맞지 않으면 **RBF 커널** 시도 (대부분의 경우 잘 동작)
3. 시간이 충분하면 **교차 검증 + 그리드 탐색**으로 `C`, `gamma` 를 튜닝

| 상황 | 추천 |
|---|---|
| 특성이 많고(수만~수십만) 데이터도 많음 | 선형 SVM |
| 데이터가 적고 특성도 적음 | RBF 커널 |
| 라벨 간 중요도가 동일 | 선형 분류가 나을 수도 있음 |

> [!note] 커널 SVM의 계산 비용
> 훈련 샘플 수가 커지면 커널 SVM은 매우 느려짐. 특성 수가 많거나 훈련 샘플이 많으면 `LinearSVC` 나 `SGDClassifier` 가 현실적

## 정리

| 개념 | 한 줄 정리 |
|---|---|
| 비선형 SVM | 다항 특성 추가 또는 커널로 비선형 경계를 만듦 |
| 커널 트릭 | 특성을 실제로 추가하지 않고 고차원 효과를 얻음 |
| RBF(가우스) 커널 | 랜드마크와의 유사도로 특성 생성, γ가 영향 범위 |
| C / γ | 둘 다 규제 성격. 크면 과대적합, 작으면 과소적합 |
