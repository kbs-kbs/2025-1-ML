# 2025-1-ML

전통적 프로그래밍: 규칙으로 데이터 생성
머신러닝: 데이터세트로 규칙 생성
머신러닝을 사용해야 하는 경우: 규칙이 길고 복잡하고 유지 보수가 잦을 때
데이터 마이닝: 인간이 인지 못한 규칙을 발견
지도 학습: 데이터에 레이블 있음
비지도 학습: 데이터에 레이블 없음
강화 학습: 데이터 없음
온라인 학습: 데이터 수집과 동시에 학습
오프라인 또는 배치 학습: 데이터 수집 단계와 학습 단계 분리
온라인 학습의 과정:
1. 손실 함수에 모델(식)을 대입한 후 모델의 각 파라미터에 대해 편미분
2. 각 편미분 함수에 수집한 데이터를 대입하여 기울기 리스트 생성
3. 기울기 리스트를 최적화 함수에 대입하여 


1. 데이터 입력
2. 데이터와 모델(식)을 손실 함수의 인자로 전달
3. 모델의 각 파라미터에 대해 손실 함수 편미분
4. 각 편미분 함수에 데이터를 대입하여 기울기 리스트 생성
5. 파라미터 리스트와 기울기 리스트를 최적화 함수의 인자로 전달
6. 새로운 파라미터 리스트 생성 후 반영

명시적 프로그래밍이 좋은 예시   
머신러닝 프로그래밍이 좋은 예시

비지도 학습 좋은 예: 지도학습하기 전에 라벨이 없을때

학습률이 높으면 - 시스템이 데이터에 빠르게 적응하지만 예전 데이터를 금방 잊음
코로나

학습률이 낮으면 - 시스템의 관성이 더 커져서 더 느리게 학습. 하지만 새로운 데이터에 있는 잡음이나 대표성 없
암

