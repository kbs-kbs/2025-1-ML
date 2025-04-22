# 2025-1-ML

- 전통적 프로그래밍: 규칙으로 데이터 생성
- 머신러닝: 데이터세트로 규칙 생성
- 머신러닝을 사용해야 하는 경우: 규칙이 길고 복잡하고 유지 보수가 잦을 때
- 데이터 마이닝: 인간이 인지 못한 규칙을 발견
- 지도 학습: 데이터에 레이블 있음 - 스팸 필터
- 비지도 학습: 데이터에 레이블 없음 - 계층 군집
- 강화 학습: 데이터 없음
- 온라인 학습: 데이터 수집과 학습을 번갈아 수행
- 오프라인 또는 배치 학습: 데이터 수집 단계와 학습 단계 분리
- 외부 메모리 학습: 외부 저장소의 데이터를 이용한 배치 학습 (데이터 수집과 학습이 비동기적)
- 사례 기반 학습: 예측 단계에서도 데이터가 필요
- 모델 기반 학습: 예측 단계에서 데이터 필요하지 않음
- 온라인 학습 과정:
  1. 손실 함수에 모델(식)을 대입한 후 모델의 각 파라미터에 대해 편미분
  2. 각 편미분 함수에 수집한 데이터를 대입하여 기울기 리스트 생성
  3. 기울기 리스트를 최적화 함수에 대입하여 새로운 파라미터 리스트 생성
  4. 파라미터 반영
- 배치 학습 과정:
  1. 손실 함수에 모델(파라미터에 대한 식)을 대입한 후 모델의 각 파라미터에 대해 편미분
  2. 각 편미분 함수에 수집한 데이터를 대입하여 기울기 리스트 생성
  3. 배치의 각 데이터에 대한 기울기 리스트의 평균 생성
  4. 기울기 평균 리스트를 최적화 함수에 대입하여 새로운 파라미터 리스트 생성
  5. 파라미터 반영

> [!note]
> 배치 학습에서는 iii.이 추가됩니다.

> [!note] 
> - 손실 함수의 x축은 모델(파라미터에 대한 식) y축은 오차     
> - 손실 함수의 기울기가 0인 지점을 최소값이라 가정     
> - 현재 모델에서의 손실 함수의 기울기가 클수록 많은 학습이 요구됨을 의미    
> - 기울기를 구할 때 각 파라미터에 대해 편미분을 해서 파라미터별로 요구되는 학습의 벡터(기울기) 산출   
> - 파라미터에 대한 기울기가 양수일 때, 손실 함수의 기울기가 0인 지점에 모델이 가까워지기 위해서는 해당 파라미터를 줄여야 하고 음수일 때에는 해당 파라미터를 키워야 함   
> - 따라서 새로운_파라미터 = 기존_파라미터 + (-(기울기)) * 학습률(하이퍼파라미터)    


- 학습률이 높은 경우: 새로운 데이터에 민감, 예전 데이터의 관성 저하
- 학습률이 낮은 경우: 새로운 데이터에 둔감, 예전 데이터의 관성 증가


- 머신러닝 파이프라인 주요 단계:
  1. 데이터 수집
  2. 데이터 전처리: 이상치/특이치 탐지
  3. 특성 공학(특성 추출, 특성 생성, 특성 선택, 특성 변환(정규화, 스케일링, 차원 축소))
  5. 시각화 (어떤 단계에서도 활용 가능)
  6. 모델 훈련(학습)
  7. 모델 평가
  8. 실제 응용/예측

- 사이킷런 프로젝트 과정 요약:
  1. 데이터 분석
  2. 모델 선택
  3. 모델 훈련
  4. 예측

## 머신러닝의 주요 도전 과제
1. 충분하지 않은 양의 훈련 데이터: 데이터가 적을수록 성능 감소
   - 해결책: 데이터 수집
2. 대표성 없는 훈련 데이터: 일반화에 필요없는 샘플이 포함, 샘플링 편향
   - 해결책: 리샘플링
3. 낮은 품질의 데이터: 특성에 이상치가 존재
   - 해결책: 샘플 제거
4. 관련없는 특성
   - 해결책: 특성 공학
5. 과대적합: 모델이 훈련 데이터에는 잘 맞지만 일반성이 떨어짐
   - 해결책: 규제 강화(하이퍼파라미터 조정), Dropout, Early Stopping
6. 과소적합: 모델이 너무 단순
   - 해결책: 규제 완화, 특성 추가
  
## 머신러닝 프로젝트 과정
1. 큰 그림:
   - 문제 정의: 주택 가격 예측
   - 학습 방식: 레이블이 있으므로 지도 학습, 시간에 따른 데이터의 변화가 적고 메모리가 충분하므로 배치 학습
   - 성능 지표(손실 함수): 벡터 사이의 거리를 재는 RMSE(평균 제곱근 오차) 또는 MAE(평균 절대 오차)
   - 가정 검사: 요구되는 출력 데이터가 숫자형인가 아니면 범주형인가? 범주형이라면 회귀가 아닌 분류
2. 데이터 구하기:
   - `housing.head()`
   - `housing.info()`: 'Non-Null Count'를 통해 특성을 가지고 있지 않은 행과 이상치를 알 수 있음
   - `housing["ocean_proximity"].value_counts()`: 범주형 특성의 범주 개수 확인
   - `housing.describe()`: 숫자형 특성의 요약 정보(표준 편차, 백분위수)
   - `housing.hist()`: 이 부분 자세히
   - 테스트 세트 생성: 데이터 스누핑 편향(성능 검증시 실제 성능보다 평가가 높아지는 현상)을 방지하기 위함
   - 이 부분도 중요한지 강의 다시
   - 테스트 세트 생성: 성능 평가의 공정성, 실험의 재현성, 실무의 신뢰성을 위해 테스트 세트를 항상 동일하게 또는 저장해서 사용하는 것이 중요
     - `train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)`
    - 계층적 샘플링: 범주형 특성의 비율을 학습/테스트 세트에서 유지하는 것
      - `train_set, test_set = train_test_split(housing, stratify=housing["income_cat"] test_size=0.2, random_state=42)`
      - 또는 `splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42); for train_index, test_index in splitter.split(housing, housing["income_cat"]):`
      - 위의 코드의 'n_splits'는 k-fold 교차 검증 시 유용
      - 숫자형 특성에서도 샘플링 편향을 방지하려면 범주형으로 변환한 후 계층적 샘플링을 하면 됨
      - `housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])`
      - 행 삭제: `set_.drop("income_cat", inplace=True)` (inplace True 는 원본에 적용 False는 바뀐 데이터프레임 반환)
      - 데이터프레임 복사: `housing = strat_train_set.copy()`
3. 데이터 시각화
   - 상관관계 조사: 특성 공학에 활용
   ```
   corr_matrix = housing.corr(numeric_only=True)
   corr_matrix["median_house_value"]
   ```
4. 데이터 정제
   - 특성에 값이 없는 경우 행 제거, 특성 제거, 값을 대체할 수 있음
   - SimpleImputer는 결측값(NaN 등)이 있는 데이터를 평균, 중앙값, 최빈값 등으로 자동 채워주는 전처리 도구 (숫자형 데이터에서만 작동)
     ```
     imputer = SimpleImputer(strategy='median')
     imputer.fit_transform(housing_num) # 중앙값 저장(fit) + 채워 넣기(transform)
     ```
   - Ordinal Data(이산적이고 순서가 있는 데이터)를 학습시키기 위해 숫자형 데이터로 변환해주어야 함 `OrdinalEncoder` 사용
     ```
     ordinal_encoder = OrdinalEncoder()
     housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
     ```
   - Categorical(Nominal) Data(이산적이고 순서가 없는 데이터)도 마찬가지로 숫자형 데이터로 변환
     ```
     one_hot_encoder = OneHotEncoder()
     housing_cat_encoded = one_hot_encoder.fit_transform(housing_cat)
     ```
     또는
     ```
     pd.get_dummies(df_test)
     ```
   - 방사 기저 함수: 위도와 경도를 하나의 특성으로 합칠 때
5. 모델 훈련
   ```
   lin_reg = make_pipeline(preprocessing, LinearRegression())
   lin_reg.fit(housing, housing_labels)
   ```
6. 예측
   ```
   housing_predictions = lin_reg.predict(housing)
   ```
   housing_predictions와 housing_labels를 비교
   - 성능 측정: `mean_squared_error(housing_labels, housing_predictions, squared=False)`
   - 교차 검증으로 평가: `tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)`
   - 그리드 서치
     ```
     grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
     grid_search.fit(housing, housing_labels)
     pd.Dataframe(grid_search.cv_results_)
     ```
   - 랜덤 서치: 모든 조합 시도하는 대신 랜덤하게
     ```
     rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
     rnd_search.fit(housing, housing_labels)
     ```
6. 모델 조정 솔루션
7. 서비스

---
## 분류
  - 오차 행렬: `cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)` cv는 폴드의 개수
  - `confusion_matrix(y_train_5, y_train_pred)`: array([[TN, FP], [FN, TP])
  - 틀린걸 틀렸다고 한것: TN 틀린걸 맞다고 한것: FP
  - 정밀도: TP/TP+FP(긍정한 것 중에서 맞은 비율) `precision_score()`
  - 재현율: TP/TP+FN(긍정인 것 중에서 맞은 비율) `recall_score()`
  - F_1 점수: 정밀도와 재현율의 조화 평균 `f1_score(y_train_5, y_train_pred)`
  - 결정 함수를 사용하여 임계값 조정 -> 높이면 정밀도 올라감, 낮추면 재현도 올라감
  - ROC 곡선: `fpr, tpr, thresholds = roc_curve(y_train_5, y_train_pred)`
  - 다중 분류: 범주형을 분류하는 전략
    1. OvR(OvA)전략: 결정 점수가 가장 높은것
    2. OvO 전략: 조합별로 결과 낸 뒤 다수결
  - 정규화: `nomalize=true`
  - ClassifierChain: 다중 레이블 분류에서 각 라벨별 예측기를 순서대로 연결해, 앞선 라벨의 예측 결과를 뒤의 라벨 분류기 입력에 추가하여 라벨 사이의 상관관계를 효과적으로 활용하는 방법

## SVM
선을 그어 분리
- 라지 마진 분류(하드, 소프트)
- 하드: 이상치 하나도 없게
- 소프트: 이상치 허용
- 하이퍼 파라미터 C: 마진 오류와 결정 경계의 간격을 조절하는 역할 C가 클수록 중립지대가 줄어듦
  c가 작을 수록 중립지대 커져서(규제 강화) 과적합을 방지할 수 있음
- degree: 높을수록 변곡점이 많아짐
- RBF: 등고선을 활용한 분리 (SVC; 커널 트릭)
- 감마: 높을 수록 울타리 쳐짐
- LinearSVC: 가장 빠르고 외부메모리 학습 지원 x, 스케일 조정은 모두 필요, 커널 트릭 x
- SVR: 회귀 버전
- 회귀에서는 엡실론도 함께 도로폭 조절에 쓰임
- 
