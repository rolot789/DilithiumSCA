# DSCAD 데이터셋을 활용한 Dilithium 부채널 공격 프로젝트

## 프로젝트 개요

 CRYSTALS-Dilithium 전자서명 표준에 대한 딥러닝 기반 프로파일링 부채널 공격(Profiled Side-Channel Attack) 연구를 위한 가이드라인입니다. DSCAD(Dilithium Side-Channel Attacks Dataset)를 활용하여 전력 분석을 통해 비밀키를 복구하는 것을 목표로 합니다.

## 데이터셋 구조

### Profiling 데이터 (모델 훈련용)
- `profiling_40000traces_set1~4.npy`: 총 160,000개의 전력 트레이스 (4개 파일 분할)
- `profiling_40000_u=cs.npy`: 레이블 데이터 (u = c·s 연산 결과)
- **특이사항**: Dilithium의 q는 8,380,417(약 23비트)이므로 직접적인 분류는 불가능하며, Hamming Weight(HW) 모델 적용이 필수적입니다.

### Attack 데이터 (키 복구 검증용)
- `attack_10000traces.npy`: 10,000개의 공격용 전력 트레이스
- `attack_s.npy`: 실제 비밀키 (검증용 정답)

### 분석 데이터
- `rho.npy`: Pearson 상관계수 (시간 샘플별 정보 누설 강도)

---

## Phase 1: 데이터 전처리 및 전략

### 1.1 POI (Point of Interest) 추출
- **전략**: |ρ| > threshold 인 지점을 기반으로 추출
- **MLP용**: 상관계수 상위 **50~200개**의 독립적인 샘플 포인트 선택
- **CNN용**: 높은 상관관계를 보이는 지점을 중심으로 **50~100 샘플 크기의 윈도우**를 추출하여 로컬 특징(Temporal Dependency) 보존
- **주의**: Montgomery Reduction(MR) 전후의 HW 누설 지점이 다를 수 있으므로, 두 지점을 모두 포함하도록 분석 필요

### 1.2 정규화 및 스케일링
- **최우선 전략**: **트레이스별(Horizontal) Z-score 정규화**
    - `(trace - trace_mean) / trace_std`
    - 측정 장비의 Gain 변화 및 시간에 따른 Baseline Shift 제거에 가장 효과적
- **고주파 노이즈 제거**: Moving Average Filter (윈도우 크기 3~7) 적용 검토

### 1.3 레이블 처리 (HW 모델링)
- **모델 선택**: $u = c \cdot s \pmod q$ 값의 **Hamming Weight (0~32)**를 레이블로 사용
- **클래스 불균형 대응**: 
    - 이항 분포 특성상 중앙값(10~16)에 데이터가 집중됨
    - **Weighted Cross-Entropy** 또는 **Label Smoothing**을 적용하여 희소 클래스 학습 강화
- **희소 클래스 처리**: 샘플 수가 너무 적은(예: 2개 미만) 클래스는 `train_test_split` 시 오류를 유발하므로, 데이터셋에서 해당 샘플을 사전에 필터링하거나 제외하는 로직 적용
- **검증**: `swar()` 함수를 통한 HW 변환 레이블의 정확성 사전 확인

### 1.4 데이터 증강 (Augmentation)
- **Noise Injection**: 실제 측정 환경을 모사하기 위해 SNR을 고려한 가우시안 노이즈 추가
- **Time Shifting**: CNN 모델의 경우 ±5~10 샘플의 미세 이동을 통해 Translation Invariance 확보

---

## Phase 2: 모델 학습 로드맵

### Step 1: 탐색적 데이터 분석 (EDA)
- `profiling_40000_u=cs.npy`의 레이블 히스토그램 시각화
- `rho.npy`를 활용하여 $u = c \cdot s$ 연산이 발생하는 정확한 시간대 특정

### Step 2: 베이스라인 수립 (Template Attack)
- 고전적 통계 공격인 Template Attack을 수행하여 "평균 몇 개의 트레이스로 키 복구가 가능한지" 기준치(Baseline) 확보

### Step 3: MLP 모델링 (빠른 검증)
- 상위 POI 100~200개를 입력으로 하는 3-4 Layer MLP 설계
- BatchNorm과 Dropout(0.2)을 활용하여 과적합 방지

### Step 4: CNN 모델링 (성능 극대화)
- **아키텍처**: 1D-CNN (Kernel Size: 11 이상 추천)
- **목표**: 필터가 전력 파형 내의 HW Transition 패턴을 스스로 학습하도록 유도
- **Global Average Pooling**: Flatten 대신 사용하여 파라미터 수 감소 및 일반화 성능 향상

### 2.5 모델 관리 (Serialization & Checkpointing)
- **목표**: 학습된 모델의 재사용성을 높이고, 최적의 상태를 영구 저장하여 성능 저하 및 중단에 대비합니다.
- **Model Checkpointing**:
    - 학습 중 `val_loss` 기준 최적의 가중치를 자동 저장 (`ModelCheckpoint` 콜백 사용)
    - 파일명에 epoch 및 성능 지표를 포함하여 버전 관리 (예: `best_model_epoch_{epoch:02d}.h5`)
- **Model Serialization**:
    - 전체 아키텍처와 학습된 가중치를 포함하는 `.h5` 또는 SavedModel 형식으로 저장
    - 모델 로드(`tf.keras.models.load_model`)를 통해 별도의 공격 파이프라인에서 즉시 재사용

---

## Phase 3: 분석 및 인사이트

### 3.1 공격 성능 지표
- **Guessing Entropy (GE)**: 공격 트레이스 수 증가에 따른 정답 키의 평균 순위 하락 분석
- **Success Rate (SR)**: 특정 트레이스 수 내에서 키를 100% 복구할 확률

### 3.2 모델 해석 (Explainable AI)
- **Grad-CAM / Integrated Gradients**: 모델이 실제로 전력 파형의 어느 부분(연산 단계)을 보고 판단하는지 시각화
- **상관관계 비교**: 통계적 PO이와 모델의 Attention 지점이 일치하는지 분석

---

## 예상되는 Challenge 및 해결책

1. **메모리 부족 (OOM)**: 16만 개의 트레이스는 매우 방대함
    - **해결**: `numpy.memmap`을 사용하거나, POI만 추출한 경량 데이터셋을 별도로 생성하여 학습
2. **학습 정체**: HW 레이블의 모호성으로 인한 Loss 수렴 지연
    - **해결**: Learning Rate Scheduler (ReduceLROnPlateau) 적용 및 최적화 함수(Adam) 사용
3. **과적합 (Overfitting)**: 프로파일링 환경에만 특화된 모델링
    - **해결**: 다양한 하이퍼파라미터 튜닝 및 강력한 규제(Regularization) 적용