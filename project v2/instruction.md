<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 🛡️ DSCAD 기반 Dilithium 부채널 공격 연구: 종합 가이드라인

## 📋 목차

1. [프로젝트 개요](#%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)
2. [데이터셋 구조 및 특성](#%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EA%B5%AC%EC%A1%B0-%EB%B0%8F-%ED%8A%B9%EC%84%B1)
3. [Phase 1: 데이터 전처리 전략](#phase-1-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%EC%A0%84%EB%9E%B5)
4. [Phase 2: 모델 설계 및 학습](#phase-2-%EB%AA%A8%EB%8D%B8-%EC%84%A4%EA%B3%84-%EB%B0%8F-%ED%95%99%EC%8A%B5)
5. [Phase 3: 공격 평가 및 인사이트](#phase-3-%EA%B3%B5%EA%B2%A9-%ED%8F%89%EA%B0%80-%EB%B0%8F-%EC%9D%B8%EC%82%AC%EC%9D%B4%ED%8A%B8)
6. [Challenge 및 해결책](#challenge-%EB%B0%8F-%ED%95%B4%EA%B2%B0%EC%B1%85)
7. [타 연구 대비 개선 포인트](#%ED%83%80-%EC%97%B0%EA%B5%AC-%EB%8C%80%EB%B9%84-%EA%B0%9C%EC%84%A0-%ED%8F%AC%EC%9D%B8%ED%8A%B8)
8. [실행 로드맵](#%EC%8B%A4%ED%96%89-%EB%A1%9C%EB%93%9C%EB%A7%B5)

***

## 프로젝트 개요

### 연구 목표

NIST 포스트양자 암호 표준인 CRYSTALS-Dilithium 전자서명에 대한 **딥러닝 기반 프로파일링 부채널 공격(Profiled Power Analysis Attack)**을 수행하여:

- Dilithium의 실제 하드웨어 구현에서 비밀키 복구 가능성을 검증
- 전력 소비 패턴과 비밀키 간의 정보 누설 메커니즘을 정량적으로 분석
- 효과적인 대응책(countermeasures) 개발을 위한 기반 마련


### 공격 시나리오

**타겟 연산**: Dilithium 서명 생성 알고리즘의 `u = c·s (mod q)` 연산

- **c**: Challenge 다항식 (공개 정보)
- **s**: 비밀키 다항식 계수 (타겟)
- **q**: 8,380,417 (약 23비트 모듈러스)


### 연구 의의

- **보안성 검증**: NIST 표준화된 Dilithium의 실제 구현 취약점 분석
- **대응책 개발**: 마스킹, 셔플링 등 대응책의 효과성 정량화
- **학술적 기여**: 포스트양자 암호에 대한 부채널 공격 방법론 확립

***

## 데이터셋 구조 및 특성

### Profiling 데이터 (모델 훈련용)

| 파일명 | 용도 | 크기 |
| :-- | :-- | :-- |
| `profiling_40000traces_set1~4.npy` | 전력 트레이스 | 160,000개 (4세트 분할) |
| `profiling_40000_u=cs.npy` | 레이블 (`u=c·s` 값) | 160,000개 |

**데이터 특성**:

- 각 트레이스는 약 40,000 샘플 포인트로 구성 (시간축)
- 샘플링 레이트와 연산 시간을 고려한 충분한 시간 해상도 제공
- 레이블은 3차원 구조로 추정: `[trace_idx, coefficient_idx, polynomial_idx]`
- **중요**: `u` 값의 범위가 0~8,380,416으로 직접 분류 불가능 → **Hamming Weight 모델링 필수**

**데이터 크기 고려사항**:

- 전체 메모리 요구량: 160,000 × 40,000 × 4bytes ≈ 25.6GB
- POI 추출 후: 160,000 × 400 × 4bytes ≈ 256MB (100배 감소)
- 메모리 효율적 처리 전략 필수


### Attack 데이터 (키 복구 검증용)

| 파일명 | 용도 |
| :-- | :-- |
| `attack_10000traces.npy` | 공격용 전력 트레이스 (10,000개) |
| `attack_10000_u=cs.npy` | 공격용 레이블 |
| `attack_10000_c.npy` | Challenge 값 |
| `attack_s.npy` | 실제 비밀키 (검증용 Ground Truth) |

**용도**:

- 학습된 모델의 키 복구 성능 평가
- Guessing Entropy 곡선 생성
- 실제 공격 시나리오 시뮬레이션


### 분석 데이터

| 파일명 | 용도 |
| :-- | :-- |
| `rho.npy` | Pearson 상관계수 (각 시간 샘플의 정보 누설 강도) |

**활용 방법**:

- POI 선택의 기초 자료
- 통계적 공격(Template Attack)의 출발점
- 딥러닝 모델의 Attention과 비교하여 해석 가능성 확보


### 데이터 구조 관련 확인 필요 사항

**질문 1**: `profiling_40000_u=cs.npy`의 정확한 차원 구조

- 특정 계수를 타겟으로 하는 경우 인덱싱 방법 확인 필요
- 다항식의 어느 계수를 공격 대상으로 선택할지 결정 필요

**질문 2**: `rho.npy`의 차원

- 단일 계수에 대한 1D 배열 `(40000,)` 인지
- 여러 계수에 대한 2D 배열 `(256, 40000)` 인지 확인

***

## Phase 1: 데이터 전처리 전략

### 1.1 POI (Points of Interest) 선택

#### 타 연구의 POI 선택 방법론

**방법 1: Correlation-based Selection (가장 일반적)**

- Pearson 상관계수의 절댓값이 높은 상위 N개 지점 직접 선택
- 성공 사례: 200-700개 POI로 SOTA 성능 달성
- 장점: 통계적으로 검증됨, 계산 비용 낮음, 해석 가능성 높음
- 구현: `rho.npy`를 로드하여 절댓값 기준 정렬 후 상위 N개 선택

**방법 2: SNR (Signal-to-Noise Ratio)**

- 클래스 간 분산(signal) / 클래스 내 평균 분산(noise) 비율
- 장점: 분류 문제에 최적화된 지표
- 단점: Pearson 상관계수 대비 계산 비용 높음
- 적용: Pearson 방법과 비교하여 성능 차이 검증

**방법 3: SOST (Sum of Squared T-differences)**

- T-test 통계량의 제곱 합
- 다중 클래스 문제에 특화
- Hamming Weight 분류에 적합할 수 있음

**방법 4: AutoPOI (Deep Reinforcement Learning)**

- DRL 에이전트가 자동으로 최적 POI 조합 탐색
- 장점: 인간의 편향 없는 선택, 때로 수동 선택보다 우수
- 단점: 계산 비용 매우 높음, 본 프로젝트에서는 over-engineering
- 향후 확장 가능성: 장기 연구 과제로 고려


#### 우리 프로젝트의 POI 전략

**MLP용: Sparse POI (독립 샘플)**

- 목표 개수: 300-500개
- 선택 방법: Pearson 상관계수 절댓값 기준 상위 N개 직접 선택
- 이유: 전역적 정보 포착, 계산 효율성, 과적합 방지
- 특징: 시간축 상에서 분산되어 있어 다양한 연산 단계 커버

**CNN용: Windowed POI (연속 구간)**

- 목표 개수: 50-100개 피크 × 윈도우 크기(11-15) = 550-1,500 샘플
- 선택 방법: 상관계수가 높은 피크를 중심으로 양방향 확장하여 윈도우 추출
- 이유: 시계열 패턴(Temporal Dependency) 보존, CNN의 Convolution 연산에 최적
- 특징: 인접 샘플 간 관계를 학습하여 전력 파형의 미세한 변화 감지

**POI 선택 시 주의사항**:

- Montgomery Reduction 전후로 전력 누설 패턴이 다를 수 있음
- NTT 연산의 여러 단계(시작, 중간, 끝)를 골고루 포함하도록 범위 확장
- 너무 많은 POI는 노이즈 증가 및 과적합 유발 (1,500개 이상 지양)
- 너무 적은 POI는 정보 손실 (200개 미만 지양)

**실험 계획**:

- Phase 1: 400개 POI로 시작 (베이스라인)
- Phase 2: 300개, 500개, 700개로 변화시키며 성능 비교
- Phase 3: 최적 POI 개수 결정 후 고정

***

### 1.2 정규화 및 스케일링 전략

#### 🚨 치명적 오류: 개별 트레이스 정규화 금지

**절대 금지되는 방법: Horizontal Normalization (트레이스별 정규화)**

- 각 트레이스의 평균과 표준편차로 해당 트레이스만 정규화
- 문제의 본질: 부채널 공격의 핵심은 서로 다른 데이터를 처리할 때 발생하는 **트레이스 간 전력 소비 차이**를 감지하는 것
- 개별 정규화 시: 모든 트레이스가 평균 0, 표준편차 1이 되어 트레이스 간 비교 불가능
- 결과: 판별 정보 완전 소실, 모델 학습 실패, 정확도 20% 근처 정체

**비유로 이해하기**:

- 학생들의 시험 점수를 비교하려는데, 각 학생의 점수를 그 학생의 평균으로 나누면 모두 "1"이 됨
- 키를 비교하려는데, 각 사람의 키를 그 사람 자신의 키로 나누면 모두 "1"이 됨
- **상대적 차이가 사라지므로 비교가 무의미해짐**


#### ✅ 올바른 정규화: Vertical Z-score Normalization

**개념**:

- 전체 Profiling 데이터셋에서 각 시간 샘플별(세로 방향)로 평균과 표준편차 계산
- 모든 트레이스를 이 전역 통계로 정규화
- 트레이스 간의 상대적 전력 차이 보존

**단계별 절차**:

1. 전체 Profiling 데이터셋 로드 (160,000 트레이스)
2. 각 시간 샘플 t에 대해 전체 트레이스의 평균과 표준편차 계산
    - `mean_per_sample[t]` = 40,000번째 샘플에서 160,000개 트레이스의 평균
    - `std_per_sample[t]` = 40,000번째 샘플에서 160,000개 트레이스의 표준편차
3. 모든 Profiling 트레이스를 동일한 통계로 정규화
4. **중요**: Attack 트레이스도 Profiling 단계에서 계산한 통계 사용 (절대 Attack 데이터의 통계 사용 금지)

**물리적 의미**:

- 특정 시간 t에서 전력이 다른 트레이스보다 높다/낮다는 절대적 정보 보존
- 측정 장비의 Gain 변화, Baseline Drift, 온도 변화 등 외부 요인 제거
- 데이터 의존적 전력 소비 차이만 강조

**타 연구의 검증**:

- 대부분의 성공적인 부채널 공격 논문에서 표준으로 사용
- Vertical 정규화 사용 시 Horizontal 대비 정확도 40-70% 향상 보고


#### 고급 기법: Data Power Trace (DPT)

**전력 소비 모델**:

```
Total_Power = P_const + P_op + P_data + P_noise
```

- `P_const`: 상수 전력 (클럭, 메모리 등)
- `P_op`: 연산 의존적 전력 (항상 동일한 연산 수행)
- `P_data`: 데이터 의존적 전력 (우리가 원하는 신호!)
- `P_noise`: 측정 노이즈

**DPT 추출 원리**:

- 모든 트레이스의 평균을 계산하면 `P_const + P_op`의 근사값 획득
- 각 트레이스에서 이 평균을 빼면 `P_data + P_noise`만 남음
- 신호 대 잡음비(SNR) 향상

**적용 절차**:

1. 전체 Profiling 트레이스의 평균(Mean Trace) 계산
2. 각 트레이스에서 Mean Trace 감산
3. 결과를 Data Power Trace로 사용
4. 이후 Vertical Z-score 정규화 적용

**효과 (실제 연구 결과)**:

- 동일한 정확도 달성 시 필요 트레이스 수 **3.64배 감소**
- 교차 바이트 공격(여러 계수 동시 공격) 시 **14.62배 효율 향상**
- 노이즈가 많은 환경에서 특히 효과적

**질문 3**: DPT를 적용할 때 Mean Trace 계산 시 모든 160,000개 트레이스를 사용해야 하는지, 아니면 샘플링(예: 10,000개)으로도 충분한지?

**우리 프로젝트의 정규화 전략**:

- Phase 1: Vertical Z-score만 적용 (베이스라인)
- Phase 2: DPT + Vertical Z-score 조합 (성능 비교)
- Phase 3: 두 방법의 성능 차이 정량화 및 최종 선택


#### 노이즈 제거: Moving Average Filter

**목적**: 고주파 측정 노이즈 제거, 신호 평활화

**적용 방법**:

- 윈도우 크기 3-7 샘플 중 선택 (5 권장)
- 각 샘플을 주변 샘플의 평균으로 대체
- Convolution 연산으로 효율적 구현 가능

**효과**:

- 순간적인 전압 변동(spike) 제거
- 전반적인 SNR 개선
- CNN 학습 안정성 향상

**주의사항**:

- CNN 사용 시 과도한 필터링은 오히려 성능 저하 가능
- CNN의 첫 번째 Convolution Layer가 자체적으로 필터링 역할 수행
- MLP는 필터링의 효과가 크지만, CNN은 선택적 적용

**실험 계획**:

- 필터 없음, 윈도우 3, 5, 7로 변화시키며 성능 비교
- MLP와 CNN에서 효과 차이 분석

***

### 1.3 레이블 처리 및 클래스 불균형 해결

#### Hamming Weight 모델링

**필요성**:

- 원본 레이블 `u = c·s (mod 8,380,417)`은 약 2^23 ≈ 8백만 개의 가능한 값
- 딥러닝 모델로 8백만 개 클래스 분류는 현실적으로 불가능
- 메모리, 계산 비용, 학습 시간 모두 과도

**Hamming Weight 정의**:

- 이진 표현에서 '1'의 개수
- 예: 13 (이진: 1101) → HW = 3
- 예: 255 (이진: 11111111) → HW = 8

**변환 과정**:

1. 원본 레이블 `u` 획득
2. 이진 표현으로 변환
3. '1'의 개수 카운트
4. 결과: 0~23 범위의 정수 (23비트이므로 최대 23개의 '1')

**분류 문제 정의**:

- 입력: 전력 트레이스 (POI 추출 후)
- 출력: Hamming Weight (0~23 또는 0~32)
- 손실 함수: Categorical Cross-Entropy
- 평가 지표: Accuracy, Guessing Entropy

**질문 4**: 23비트 정수의 Hamming Weight는 0~23 (24개 클래스)인데, 왜 33-class로 설정했는지?

- Dilithium의 다른 파라미터(η=2, 4)와 관련이 있는지
- 32비트 표현을 고려한 것인지
- 데이터셋 문서 확인 필요

**정보 손실 고려**:

- HW 변환 시 원본 값의 상당 부분 정보 손실
- 예: u=7 (HW=3)과 u=11 (HW=3)을 구분 불가
- 그럼에도 불구하고 HW 공격이 성공하는 이유: 여러 트레이스의 정보를 결합하면 키 복구 가능


#### 클래스 불균형 문제

**이항 분포 특성**:

- 23비트에서 Hamming Weight는 이항 분포 B(23, 0.5) 따름
- 중앙값 (HW=11, 12) 근처에 데이터 집중 (약 30-40%)
- 극단값 (HW=0, 1, 22, 23)은 매우 희소 (<0.1%)
- 이는 동전을 23번 던졌을 때 앞면이 11~12번 나올 확률이 가장 높은 것과 동일

**문제 발생**:

- 모델이 다수 클래스(중앙값)만 학습하고 소수 클래스 무시
- Validation loss는 감소하지만 accuracy는 정체 (20% 근처)
- 전체 정확도는 높아도 특정 HW 값에 대한 예측은 완전히 실패
- 결과: 실제 키 복구 시 특정 계수를 절대 맞출 수 없음


#### 클래스 불균형 해결 방법

**방법 1: Class Weights (1순위 추천)**

- 개념: 손실 함수 계산 시 희소 클래스에 더 높은 가중치 부여
- 계산: 역빈도 가중치 사용 (빈도가 낮을수록 높은 가중치)
- 구현: sklearn의 `compute_class_weight` 함수 활용
- 장점: 구현 간단, 효과적, 과적합 위험 낮음
- 단점: 극단적 불균형에서는 불충분할 수 있음

**방법 2: Focal Loss (고급)**

- 개념: 어려운 샘플(잘못 분류되는 샘플)에 더 집중
- 수식: `FL = -(1 - p_t)^gamma * log(p_t)`
- `gamma`: 집중 정도 조절 (2 권장)
- 장점: Class Weights보다 강력, 극단적 불균형 처리 가능
- 단점: 하이퍼파라미터 튜닝 필요, 수렴 느릴 수 있음

**방법 3: Label Smoothing (보조)**

- 개념: 타겟 레이블을 Hard (0 또는 1)에서 Soft (0.05 또는 0.95)로 변경
- 효과: 과신(overconfidence) 방지, 일반화 성능 향상
- 주의: 클래스 불균형 상황에서는 소수 클래스 학습을 더욱 어렵게 만들 수 있음
- 권장값: 0.05 (0.1은 너무 강함)
- 적용 시기: Class Weights와 함께 사용 시 주의 필요

**방법 4: 희소 클래스 필터링 (데이터 정제)**

- 개념: 샘플 수가 극히 적은 클래스를 데이터셋에서 제거
- 기준: 샘플 수 < 2개인 클래스 (train_test_split의 stratify 오류 방지)
- 또는: 샘플 수 < 150개인 클래스 (통계적 신뢰성 확보)
- 장점: 학습 안정성 향상, 오류 방지
- 단점: 정보 손실, 전체 키 복구 시 해당 계수는 다른 방법으로 추론 필요

**방법 5: Over-sampling / Under-sampling (일반적으로 비권장)**

- Over-sampling: 소수 클래스 복제
- Under-sampling: 다수 클래스 샘플 제거
- 문제: 부채널 데이터의 물리적 특성 왜곡, 과적합 유발
- 결론: 부채널 공격에서는 사용하지 않는 것이 일반적

**우리 프로젝트의 전략**:

- 필수: Class Weights 적용
- 보조: Label Smoothing 0.05 (선택적)
- 데이터 정제: 샘플 수 < 150개 클래스 필터링
- 평가: 클래스별 정확도 분석하여 불균형 해소 확인

***

### 1.4 데이터 증강 (Augmentation)

#### 타 연구의 Augmentation 전략

**기본 원칙**:

- 부채널 데이터는 물리적 측정값이므로 과도한 증강은 왜곡 유발
- 실제 측정 환경에서 발생 가능한 변화만 시뮬레이션
- CNN에만 제한적으로 적용 (MLP는 불필요)

**방법 1: Gaussian Noise Injection (권장)**

- 목적: 실제 측정 환경의 전자기 노이즈, 열 노이즈 모사
- 표준편차: 0.05 (정규화된 데이터 기준)
- 적용 시기: Training 시에만 (Validation/Test에는 적용 금지)
- 구현: Keras의 GaussianNoise Layer 활용
- 효과: 일반화 성능 향상, 노이즈에 강건한 모델
- 주의: stddev > 0.1은 신호 왜곡 유발, 성능 저하

**방법 2: Time Shifting (신중하게 적용)**

- 목적: 트레이스 정렬(alignment) 오차에 대한 강건성 확보
- 범위: ±5 샘플 정도의 랜덤 이동
- 효과: Translation invariance 향상
- **위험성**:
    - POI 위치가 변하므로 공격 성능 저하 가능
    - 특히 Windowed POI 사용 시 핵심 정보가 윈도우 밖으로 벗어날 수 있음
    - 실제 측정 환경에서 정렬은 보통 잘 되어 있음
- **결론**: 사용하지 않는 것을 권장

**방법 3: Amplitude Scaling (효과 미미)**

- 목적: 측정 장비의 Gain 변화 시뮬레이션
- 범위: 0.95~1.05 배 랜덤 스케일링
- 문제: Vertical 정규화를 사용하면 이미 Gain 변화가 보정됨
- 결론: 불필요

**방법 4: Adding Multiple Traces (고급)**

- 개념: 서로 다른 두 트레이스를 선형 결합
- 예: `new_trace = 0.7 * trace_A + 0.3 * trace_B`
- 효과: 데이터셋 크기 확장, 일반화 성능 향상
- 문제: 물리적 의미가 불분명, 레이블 처리 복잡
- 결론: 실험적 시도 가능하지만 우선순위 낮음

**타 연구 결론**:

- 대부분의 성공적인 부채널 공격 논문에서 Gaussian Noise 0.05만 사용
- 과도한 Augmentation은 오히려 성능 저하
- 데이터의 물리적 특성 유지가 최우선

**우리 프로젝트의 Augmentation 전략**:

- CNN 모델: GaussianNoise(0.05) 적용
- MLP 모델: Augmentation 없음
- Time Shifting: 사용하지 않음
- 실험: Noise stddev를 0, 0.03, 0.05, 0.07로 변화시키며 성능 비교

***

## Phase 2: 모델 설계 및 학습

### 2.1 베이스라인: Template Attack

#### 개념 및 목적

**Template Attack이란**:

- 고전적 통계 기반 프로파일링 부채널 공격
- 가우시안 분포 가정 하에 최대 우도 추정(MLE) 사용
- 딥러닝 이전 시대의 SOTA 방법

**연구에서의 역할**:

- 딥러닝 모델 성능을 평가할 기준점(baseline) 확립
- "평균 몇 개의 트레이스로 키 복구가 가능한지" 정량화
- 딥러닝의 성능 향상 정도를 상대적으로 측정


#### 알고리즘 원리

**Profiling 단계**:

1. 각 Hamming Weight 클래스 c에 대해 해당 클래스의 모든 트레이스 수집
2. 평균 벡터 계산: 해당 클래스 트레이스들의 평균
3. 공분산 행렬 계산: 클래스 내 변동성 측정
4. 결과: 각 클래스마다 다변량 가우시안 분포의 파라미터 저장

**Attack 단계**:

1. 새로운 공격 트레이스 획득
2. 각 클래스의 가우시안 분포에서 해당 트레이스가 나올 우도 계산
3. 가장 높은 우도를 가진 클래스 선택
4. 여러 트레이스 사용 시: 독립 가정 하에 우도를 곱함 (또는 로그 우도를 합산)

**성능 지표**:

- 단일 트레이스 정확도
- N개 트레이스 결합 시 Guessing Entropy
- 전체 키 복구에 필요한 평균 트레이스 수


#### 타 연구의 Template Attack 성능

**Dilithium 공격 결과**:

- Unprotected 구현: 단일 계수 복구에 130개 트레이스 필요
- 단일 트레이스 기준 정확도: 60-70%
- 전체 키 복구: 약 5,000개 트레이스

**AES 공격 결과 (비교 참고)**:

- ASCAD 데이터셋: 50-100개 트레이스로 키 복구
- 단일 트레이스 정확도: 40-60%

**한계점**:

- 고차원 데이터에서 공분산 행렬 추정 어려움
- 가우시안 분포 가정이 항상 성립하지 않음
- POI 선택에 매우 민감

**우리 프로젝트의 목표**:

- Template Attack으로 베이스라인 성능 측정
- 딥러닝 모델이 최소 Template Attack 성능의 1.5배 이상 달성
- 예상: Template Attack 70% 정확도 → 딥러닝 85% 이상

***

### 2.2 MLP (Multi-Layer Perceptron)

#### 아키텍처 설계 원칙

**기본 구조**:

- Input Layer: POI 개수 (300-500)
- Hidden Layers: 2-4개 (256-512 뉴런)
- Output Layer: 클래스 수 (24 또는 33)

**레이어 구성**:

- Dense(256, relu) + BatchNormalization + Dropout(0.2)
- Dense(512, relu) + BatchNormalization + Dropout(0.3)
- Dense(256, relu) + Dropout(0.2)
- Dense(num_classes, softmax)

**설계 논리**:

- 첫 레이어: 입력 특징을 고차원으로 확장하여 표현력 증대
- 중간 레이어: 비선형 변환을 통해 복잡한 패턴 학습
- 마지막 레이어: 분류를 위한 차원 축소
- Dropout: 각 레이어 출력의 일부를 랜덤하게 0으로 만들어 과적합 방지
- BatchNormalization: 각 배치의 평균과 분산을 정규화하여 학습 안정화


#### 하이퍼파라미터 설정

**Optimizer**:

- 1순위: Adam (learning_rate=0.001)
- 특징: 적응적 학습률, 빠른 수렴, 하이퍼파라미터 튜닝 부담 적음
- 대안: Nadam (Nesterov momentum 추가, 약간 더 빠른 수렴)

**Batch Size**:

- DSCAD (160K 샘플): 256 권장
- 이유: 큰 배치는 gradient 추정의 분산 감소, 학습 안정성 향상
- 메모리 부족 시: 128로 감소 가능

**Epochs**:

- 최대: 100
- Early Stopping으로 실제로는 30-50 에포크에서 종료 예상

**Learning Rate Schedule**:

- ReduceLROnPlateau 사용
- patience=5 (5 에포크 동안 개선 없으면)
- factor=0.5 (학습률 절반으로 감소)
- min_lr=1e-6 (최소 학습률)


#### 예상 성능 및 장단점

**타 연구 성능**:

- ASCAD (AES): 85-90% 정확도
- Dilithium (예상): 70-80% 정확도

**장점**:

- 빠른 학습 속도 (POI 기반이므로 입력 차원 낮음)
- 하이퍼파라미터 튜닝 간단
- 해석 가능성 상대적으로 높음
- 베이스라인 확립에 최적

**단점**:

- 시계열 패턴 무시 (인접 샘플 간 관계 학습 불가)
- CNN 대비 성능 5-10% 낮음
- 공간적 특징 추출 능력 부족

**우리 프로젝트의 활용**:

- Phase 1에서 MLP로 빠르게 베이스라인 수립
- 전처리 방법(정규화, POI 선택)의 효과 검증
- CNN 모델 개발 전 데이터 품질 확인

***

### 2.3 CNN (Convolutional Neural Network)

#### CNN의 부채널 공격 적합성

**왜 CNN인가**:

- 전력 트레이스는 시계열 데이터 (1D 신호)
- 인접 샘플 간의 관계가 중요 (예: 전력 파형의 상승/하강 패턴)
- Convolutional Layer는 국소적 패턴(local pattern)을 자동으로 학습
- Translation invariance: 패턴의 위치가 약간 변해도 인식 가능

**물리적 의미**:

- 각 Conv Filter는 특정 전력 소비 패턴(예: Hamming Weight 전환 시 발생하는 파형)을 감지하는 템플릿 역할
- 여러 Filter를 통해 다양한 패턴을 동시에 포착
- Pooling Layer는 미세한 위치 변동에 강건하게 만듦


#### 타 연구의 CNN 아키텍처 동향

**아키텍처 1: Zaid's Efficient CNN (검증된 SOTA)**

- 성공 사례: Dilithium-2/3/5에서 10/27/18 트레이스로 비밀키 복구
- 구조 개요:
    - Conv1D(64 filters, kernel_size=11, stride=1, relu)
    - BatchNormalization
    - MaxPooling1D(pool_size=2)
    - Conv1D(128 filters, kernel_size=11, relu)
    - BatchNormalization
    - MaxPooling1D(pool_size=2)
    - GlobalAveragePooling1D
    - Dense(256, relu)
    - Dropout(0.4)
    - Dense(num_classes, softmax)

**핵심 설계 원칙 (Zaid Architecture)**:

1. **큰 Kernel Size (11 이상)**: 전력 파형의 넓은 수용 영역(receptive field) 확보
    - 이유: 부채널 신호는 여러 샘플에 걸쳐 나타남
    - 작은 커널(3, 5)은 국소적 특징만 포착하여 성능 저하
2. **Stride=1 유지**: 초기 레이어에서 정보 손실 방지
    - 공격적인 다운샘플링(stride=2)은 중요한 신호 누락 가능
3. **GlobalAveragePooling**: Flatten 대신 사용
    - 효과: 파라미터 수 대폭 감소, 과적합 방지
    - 각 Feature Map의 전역 정보를 평균으로 요약
4. **BatchNorm**: 모든 Conv Layer 후 적용
    - 학습 안정화, 내부 공변량 이동(Internal Covariate Shift) 방지
    - 높은 학습률 사용 가능

**아키텍처 2: ResNet 기반 (깊은 네트워크)**

- 적용 분야: 마스킹된 구현, Desynchronization 환경
- 성능: 기존 CNN 대비 5-10% 향상
- 핵심 개선 요소:
    - Residual Connection: 입력을 출력에 직접 더함 (Skip Connection)
    - 효과: 깊은 네트워크에서 기울기 소실 문제 해결, 저수준 특징 보존
    - Identity Mapping: 학습이 필요 없으면 입력을 그대로 통과시킬 수 있음
- 주의사항:
    - 과도한 깊이(10 블록 이상)는 과적합 유발
    - 3-4 ResNet 블록 권장
    - 계산 비용 증가

**아키텍처 3: Attention 메커니즘 (해석 가능성 증대)**

- 방법: CBAM (Convolutional Block Attention Module) 추가
    - Channel Attention: 어떤 특징 맵(filter)이 중요한지 학습
    - Spatial Attention: 어느 시간 샘플이 중요한지 학습
- 효과:
    - 모델이 자동으로 중요한 POI에 집중
    - 표준 CNN 대비 2-5% 성능 향상
    - Grad-CAM 없이도 중요 시점 파악 가능
- 단점:
    - 계산 비용 증가
    - 하이퍼파라미터 추가 (attention reduction ratio 등)
    - 해석 복잡도 증가

**아키텍처 4: Mamba/Transformer 기반 (최신 연구)**

- 구조: State Space Model 또는 Self-Attention 메커니즘
- 특징: Long-range dependency 모델링 (먼 샘플 간 관계 학습)
- 성능: ASCAD에서 GE=1 달성에 100개 트레이스 미만
- 상태: 연구 초기 단계, Dilithium 적용 사례 아직 없음
- 단점: 계산 비용 매우 높음, 과적합 위험


#### 우리 프로젝트의 CNN 전략

**Phase 1: Simplified Efficient CNN (최우선)**

- 목표: 타 연구(Zaid) 재현 및 검증
- 구조: Zaid's Efficient CNN을 정확히 구현
- 기대 성능: 85-92% 정확도
- 학습 교훈: 검증된 아키텍처의 성능을 먼저 확인

**Phase 2: Inception-ResNet 변형 (고급 실험)**

- 목표: 성능 극대화 시도
- 개선점:
    - Inception Module: 다양한 Kernel Size (3, 7, 11)를 병렬로 처리하여 다양한 스케일의 패턴 포착
    - Residual Connection: 학습 안정화 및 기울기 소실 방지
    - 예상 효과: 3-5% 추가 성능 향상
- 주의사항:
    - 복잡도 증가로 과적합 위험 증가
    - 강한 Regularization 필수 (Dropout 0.5, L2 regularization)
    - Shortcut과 Merged Branch의 차원 일치 확인 필수

**Phase 3: 성능 비교 및 최종 선택**

- Zaid CNN vs Inception-ResNet
- 학습 시간, 메모리 사용량, 최종 정확도 종합 고려
- 실용성과 성능의 균형점 찾기

**질문 5**: 현재 구현한 Inception-ResNet 블록에서 shortcut과 merged branch의 차원이 정확히 일치하는지 확인 필요

- Add 연산 시 두 텐서의 shape이 완전히 동일해야 함
- 불일치 시: Conv1D(filters, 1) 레이어로 차원 조정 필요

***

### 2.4 하이퍼파라미터 최적화

#### Optimizer 선택

**1순위: Adam**

- 학습률: 0.001 (기본값)
- 특징:
    - Adaptive Moment Estimation (1차, 2차 모멘트 모두 사용)
    - 각 파라미터마다 다른 학습률 적용
    - 빠른 수렴, 안정적 학습
- 장점: 하이퍼파라미터 튜닝 부담 적음, 대부분의 문제에서 잘 작동
- 단점: 때로 일반화 성능이 SGD보다 약간 낮을 수 있음 (부채널 공격에서는 문제 없음)

**2순위: Nadam**

- 학습률: 0.0005-0.001
- 특징: Adam + Nesterov Momentum
- 장점: Adam보다 약간 더 빠른 수렴
- 단점: 하이퍼파라미터 하나 더 추가

**사용 금지: SGD (Momentum 포함)**

- 이유: 부채널 데이터는 노이즈가 많고 loss landscape가 복잡함
- SGD는 고정 학습률 사용 시 수렴 어렵고, 학습률 스케줄링 필수
- 적응적 학습률을 사용하는 Adam 계열이 훨씬 효과적


#### Learning Rate Schedule

**ReduceLROnPlateau (필수 적용)**

- 모니터링 지표: `val_loss`
- factor: 0.5 (학습률을 절반으로 감소)
- patience: 5 (5 에포크 동안 개선 없으면 학습률 감소)
- min_lr: 1e-6 (최소 학습률, 이보다 낮아지지 않음)
- 효과:
    - Plateau(정체) 구간에서 학습률 감소로 미세 조정 가능
    - 과도하게 높은 학습률로 인한 진동 방지
    - 최종 수렴 성능 향상

**대안: CosineAnnealingLR (고급)**

- 학습률을 코사인 함수 형태로 주기적 감소
- Warm Restart 가능
- 단점: 에포크 수를 미리 정해야 함


#### Batch Size 선택

**원칙**:

- 작은 데이터셋 (<10K): 32-64
- 중간 데이터셋 (10K-100K): 64-128
- 큰 데이터셋 (>100K): 128-256

**DSCAD (160K 샘플): 256 권장**

- 이유:
    - 큰 배치는 gradient 추정의 분산(variance) 감소
    - 학습 안정성 향상
    - GPU 활용률 증가 (병렬 처리)
- 메모리 부족 시: 128로 감소 가능
- 너무 작은 배치(32 이하): 학습 불안정, 수렴 느림


#### Early Stopping

**설정**:

- 모니터링 지표: `val_loss`
- patience: 15 (15 에포크 동안 개선 없으면 학습 중단)
- restore_best_weights: True (최고 성능 가중치로 복원)

**patience 설정 논리**:

- 너무 짧으면 (5 이하): 조기 종료 위험, 학습률 감소 후 개선 가능성 차단
- 너무 길면 (30 이상): 불필요한 학습 시간 낭비
- 15: ReduceLROnPlateau와 조합 시 적절한 균형


#### Regularization 조합

**Dropout**:

- Dense Layer: 0.3-0.5
- Conv Layer 후: 보통 사용하지 않음 (BatchNorm으로 충분)
- 원리: 학습 시 뉴런의 일부를 랜덤하게 비활성화하여 뉴런 간 co-adaptation 방지
- 효과: 과적합 방지, 일반화 성능 향상

**L2 Regularization**:

- 계수: 1e-4 ~ 1e-5
- 적용: 모든 Dense, Conv Layer의 kernel_regularizer
- 원리: 가중치의 L2 norm을 손실 함수에 추가하여 큰 가중치 억제
- 효과: 모델이 특정 특징에 과도하게 의존하는 것 방지

**BatchNormalization**:

- 위치: 모든 Conv Layer 후, 활성화 함수 전
- 원리: 각 배치의 평균을 0, 분산을 1로 정규화
- 효과:
    - 학습 안정화 (Internal Covariate Shift 방지)
    - 높은 학습률 사용 가능
    - 약간의 Regularization 효과

**Label Smoothing**:

- 계수: 0.05 (0.1은 너무 강함)
- 원리: Hard Label (0, 1)을 Soft Label (0.05, 0.95)로 변경
- 효과: 과신(overconfidence) 방지, 일반화 성능 향상
- 주의: 클래스 불균형과 함께 사용 시 소수 클래스 학습을 더욱 어렵게 만들 수 있음

**Regularization 적용 우선순위**:

1. BatchNormalization (필수)
2. Dropout 0.3-0.5 (필수)
3. Early Stopping (필수)
4. L2 Regularization (선택, 과적합 심할 때)
5. Label Smoothing (선택, 신중하게)

***

### 2.5 모델 관리 (Checkpointing \& Serialization)

#### ModelCheckpoint 전략

**기본 설정**:

- 파일 경로: `models/best_model_epoch{epoch:02d}_val_loss{val_loss:.4f}.h5`
- 모니터링 지표: `val_loss`
- save_best_only: True (최고 성능만 저장)
- mode: 'min' (val_loss 최소화)
- verbose: 1 (저장 시 메시지 출력)

**저장 전략**:

1. **Best Model만 저장**: 디스크 공간 절약, 최종 평가에 사용
2. **주기적 저장** (선택): 매 5 에포크마다 저장하여 학습 과정 추적
3. **Last Model 저장** (선택): 학습이 중단되어도 재개 가능

**파일명 규칙**:

- 타임스탬프 포함: `model_20260207_183000.h5`
- 설정 해시 포함: `model_pearson400_vertical_adam.h5`
- 성능 지표 포함: `model_val_acc_0.8512.h5`


#### 메타데이터 저장 (재현성 확보)

**저장 정보**:

- 실험 일시
- 모델 아키텍처 이름
- 전처리 설정 (POI 방법, 정규화 방법, 필터 여부 등)
- 하이퍼파라미터 (배치 크기, 학습률, 옵티마이저 등)
- 학습 결과 (최종 epoch, 최고 성능 지표)
- 학습 시간, 메모리 사용량

**파일 형식**: JSON (human-readable, 파싱 쉬움)

**예시 구조**:

```
{
  "experiment_id": "exp_20260207_001",
  "timestamp": "2026-02-07 18:30:00",
  "model": {
    "architecture": "Efficient_CNN",
    "total_parameters": 1234567,
    "trainable_parameters": 1234000
  },
  "preprocessing": {
    "poi_method": "pearson_correlation",
    "poi_count": 400,
    "normalization": "vertical_zscore",
    "data_power_trace": true,
    "moving_average_window": 5
  },
  "hyperparameters": {
    "batch_size": 256,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "epochs_max": 100
  },
  "training": {
    "epochs_trained": 47,
    "early_stopping_triggered": true,
    "training_time_seconds": 3627
  },
  "performance": {
    "best_val_loss": 0.4523,
    "best_val_accuracy": 0.8812,
    "final_train_accuracy": 0.9234,
    "guessing_entropy_100traces": 5.2
  }
}
```


#### 실험 추적 도구 (선택사항)

**MLflow**:

- 장점: 실험 자동 로깅, 웹 UI 제공, 모델 레지스트리
- 단점: 초기 설정 필요

**Weights \& Biases (wandb)**:

- 장점: 실시간 시각화, 협업 기능, 클라우드 기반
- 단점: 외부 서비스 의존, 데이터 업로드 시간

**TensorBoard**:

- 장점: TensorFlow 통합, 로컬 실행
- 단점: 기능 제한적

**질문 6**: 여러 실험을 체계적으로 관리하기 위해 실험 추적 도구를 사용할 계획인지?

- 실험 개수가 많다면 (50개 이상) 도구 사용 권장
- 소규모 연구라면 수동 관리도 충분


#### 모델 버전 관리

**Git LFS (Large File Storage)**:

- 모델 파일을 Git으로 버전 관리
- 변경 이력 추적 가능
- 협업 시 유용

**폴더 구조 예시**:

```
results/
├── models/
│   ├── exp_001_best.h5
│   ├── exp_001_metadata.json
│   ├── exp_002_best.h5
│   └── exp_002_metadata.json
├── logs/
│   ├── exp_001_training.log
│   └── exp_002_training.log
├── figures/
│   ├── exp_001_learning_curve.png
│   └── exp_001_ge_curve.png
└── cache/
    ├── profiling_data_poi400.npy
    └── attack_data_poi400.npy
```


***

## Phase 3: 공격 평가 및 인사이트

### 3.1 성능 지표

#### Guessing Entropy (GE)

**정의**:

- 올바른 비밀키를 찾기 위해 평균적으로 시도해야 하는 횟수
- GE = 1: 첫 번째 추측에서 정답 (완벽한 공격)
- GE = 2^23: 랜덤 추측과 동일 (공격 실패)

**계산 방법**:

1. 각 공격 트레이스에 대해 모델이 출력한 확률 분포 획득
2. N개 트레이스를 결합: 독립 가정 하에 로그 확률을 합산
3. 합산된 로그 확률을 내림차순 정렬
4. 실제 정답 키의 순위(rank) 확인
5. 여러 공격 실험의 순위를 평균화

**해석**:

- GE가 작을수록 공격 성능 좋음
- GE vs 트레이스 수 곡선: 트레이스가 증가할수록 GE 감소
- 실용적 기준: GE < 10 (10번 이내 시도로 키 복구 가능)

**타 연구 성능 벤치마크**:

- Dilithium-2 (Zaid CNN): 10개 트레이스로 GE < 10
- Dilithium-3: 27개 트레이스로 GE < 10
- Dilithium-5: 18개 트레이스로 GE < 10
- ASCAD (AES): 100개 트레이스로 GE = 1

**우리의 목표**:

- 100개 트레이스로 GE < 5
- 500개 트레이스로 GE = 1 (완전 키 복구)


#### Success Rate (SR)

**정의**:

- N개 트레이스로 비밀키를 완전히 복구하는 확률
- GE=1 달성 비율

**계산**:

- 여러 독립적인 공격 실험 수행 (예: 100회)
- 각 실험에서 N개 트레이스 사용 시 GE=1 달성 여부 확인
- Success Rate = (GE=1 달성 횟수) / (전체 실험 횟수)

**해석**:

- SR=1.0 (100%): N개 트레이스로 항상 키 복구 가능
- SR=0.9 (90%): N개 트레이스로 10번 중 9번 성공
- SR=0.0 (0%): N개 트레이스로는 키 복구 불가능

**실용적 의미**:

- 공격자 입장: SR > 0.9이면 실용적 공격 가능
- 방어자 입장: SR > 0.1이면 취약점으로 간주


#### 단일 트레이스 정확도

**정의**:

- 하나의 트레이스로 Hamming Weight를 올바르게 분류하는 비율
- 표준 분류 문제의 정확도

**역할**:

- 모델의 기본 성능 지표
- 전처리 방법 비교 시 빠른 피드백
- GE와 높은 상관관계 (정확도 높으면 GE 낮음)

**목표 설정**:

- MLP: 75-85%
- Efficient CNN: 85-92%
- Inception-ResNet: 90-95%
- Template Attack 대비 최소 10% 이상 향상

**주의사항**:

- 클래스 불균형 시 정확도만으로는 불충분
- 클래스별 정확도 분석 필수
- Precision, Recall, F1-score 보조 지표로 활용


#### 전체 키 복구 시간

**정의**:

- Dilithium 비밀키 전체를 복구하는 데 필요한 트레이스 수 및 시간

**고려사항**:

- Dilithium-2: 256개 계수 공격 필요
- 각 계수마다 독립적으로 Guessing Entropy 계산
- 가장 어려운 계수(worst-case)를 기준으로 전체 복구 시간 추정

**타 연구 성능**:

- Unprotected Dilithium-2: 5,000-10,000 트레이스로 전체 키 복구
- 계산 시간: 일반 PC로 0.5-2시간

***

### 3.2 모델 해석 (Explainable AI)

#### Grad-CAM (Gradient-weighted Class Activation Mapping)

**목적**:

- CNN이 어느 시간 샘플(POI)을 보고 판단하는지 시각화
- "모델이 트레이스의 어디를 주목하는가?"에 대한 답

**원리**:

1. 특정 클래스에 대한 출력의 gradient를 계산
2. 마지막 Convolutional Layer의 Feature Maps에 대한 gradient 가중 평균
3. Feature Maps를 가중 평균하여 Heatmap 생성
4. Heatmap을 원본 트레이스 크기로 업샘플링

**시각화 방법**:

- 원본 트레이스 위에 Heatmap을 Overlay
- 빨간색: 모델이 주목하는 영역 (중요도 높음)
- 파란색: 무시하는 영역 (중요도 낮음)

**분석 질문**:

1. CNN의 Attention이 Pearson 상관계수가 높은 지점과 일치하는가?
    - 일치: 모델이 통계적 POI를 잘 학습
    - 불일치: 모델이 새로운 누설 지점 발견 가능성
2. NTT 연산의 어느 단계(시작/중간/끝)를 주로 보는가?
3. 다른 Hamming Weight 클래스 예측 시 주목하는 지점이 다른가?

**활용**:

- 디버깅: 모델이 노이즈나 무관한 영역을 보고 있다면 전처리 재검토
- 인사이트: 새로운 POI 발견 → 향후 공격 효율성 향상
- 논문 작성: 모델의 해석 가능성 증명


#### Integrated Gradients

**Grad-CAM 대비 장점**:

- 더 정확한 기여도 측정
- 모든 레이어에 적용 가능 (Conv뿐만 아니라 Dense도)

**단점**:

- 계산 비용 매우 높음 (여러 interpolation step 필요)
- 구현 복잡

**적용 시기**:

- Grad-CAM으로 충분하지 않을 때
- 논문 리뷰어가 더 정교한 해석을 요구할 때


#### Saliency Map

**개념**:

- 입력 트레이스의 각 샘플이 출력에 미치는 영향 계산
- Gradient의 절댓값으로 측정

**장점**:

- 구현 간단
- 직관적 해석

**단점**:

- 노이즈가 많음
- Grad-CAM이 일반적으로 더 선호됨

***

### 3.3 대응책 평가 (Countermeasure Simulation)

#### 마스킹 (Masking)

**개념**:

- 비밀값을 랜덤 마스크와 XOR하여 전력 소비 패턴 은닉
- 1차 마스킹: s' = s ⊕ r (r은 랜덤)
- 2차 마스킹: s' = s ⊕ r1 ⊕ r2

**시뮬레이션 방법**:

- 공격 트레이스에 랜덤 마스크의 전력 소비를 추가
- 마스크 갱신 빈도에 따라 효과 변화

**타 연구 결과**:

- 1차 마스킹: 필요 트레이스 2-5배 증가
- 2차 마스킹: 700개 트레이스 필요 (Unprotected 대비 70배)
- 3차 마스킹: 2,400개 트레이스 (240배)

**우리 프로젝트의 평가**:

- 1차 마스킹 효과 정량화
- 마스킹 오버헤드 (성능, 코드 크기) vs 보안 이득 분석


#### 셔플링 (Shuffling)

**개념**:

- 연산 순서를 랜덤하게 섞어 시간축 정렬(alignment) 방해
- 트레이스 간 동일 연산이 다른 시점에 발생

**효과**:

- Desynchronization으로 평균 트레이스 계산 불가능
- POI가 시간적으로 분산되어 공격 어려움

**시뮬레이션 방법**:

- 공격 트레이스에 랜덤 time shift 적용
- Shift 범위: ±50~200 샘플

**대응 공격 방법**:

- Elastic alignment: Dynamic Time Warping (DTW)
- 통계적 re-alignment
- CNN의 translation invariance 활용


#### 노이즈 레벨 변화

**목적**:

- SNR (Signal-to-Noise Ratio) 변화에 따른 공격 성능 평가
- "얼마나 노이즈가 많아야 공격이 실패하는가?"

**방법**:

- 깨끗한 트레이스에 가우시안 노이즈 추가
- SNR: 10, 20, 30, 40, 50 dB로 변화

**성능 곡선**:

- X축: SNR (dB)
- Y축: Guessing Entropy 또는 Success Rate
- 분석: SNR이 감소할수록 공격 성능 저하

**실용적 의미**:

- 방어자: 노이즈 추가로 공격 어렵게 만들기 (노이즈 생성기)
- 공격자: 저SNR 환경에서도 작동하는 강건한 모델 개발

***

## Challenge 및 해결책

### Challenge 1: 메모리 부족 (Out Of Memory)

#### 문제 상황

**메모리 요구량 계산**:

- 전체 데이터: 160,000 트레이스 × 40,000 샘플 × 4 bytes = 25.6 GB
- 일반 노트북: 8-16 GB RAM → 메모리 부족

**증상**:

- 데이터 로딩 시 MemoryError
- 학습 중 Kernel crash
- 시스템 전체 멈춤 (swap 사용으로 속도 극도로 느려짐)


#### 해결책 1: Memory-mapped I/O

**원리**:

- 파일을 메모리에 완전히 로드하지 않고, 필요한 부분만 디스크에서 읽어옴
- OS의 가상 메모리 시스템 활용

**구현**:

- NumPy의 `mmap_mode='r'` 옵션 사용
- 읽기 전용으로 파일 맵핑

**장점**:

- 메모리 사용량 대폭 감소
- 큰 파일도 즉시 접근 가능

**단점**:

- 랜덤 접근 시 디스크 I/O로 인한 속도 저하
- 순차 접근은 문제없음


#### 해결책 2: POI 사전 추출

**원리**:

- 40,000 샘플 중 400개 POI만 추출하여 별도 파일로 저장
- 25.6 GB → 0.256 GB (100배 감소)

**절차**:

1. 한 번만 전체 데이터 로드 (또는 Memory-mapped I/O 사용)
2. POI 인덱스 적용하여 필요한 샘플만 추출
3. 경량 데이터를 `.npy` 파일로 저장
4. 이후 학습에서는 경량 데이터만 사용

**장점**:

- 메모리 문제 완전 해결
- 이후 실험에서 로딩 속도 대폭 향상

**주의**:

- POI 변경 시 재추출 필요
- 디스크 공간 추가 사용 (큰 문제 아님)


#### 해결책 3: Data Generator (배치 단위 로딩)

**원리**:

- 전체 데이터를 메모리에 올리지 않고, 배치 크기만큼만 로딩
- 학습 시 배치 단위로 데이터 생성

**구현**:

- Python Generator 또는 Keras Sequence 사용
- 매 iteration마다 디스크에서 배치 읽어옴

**장점**:

- 무한히 큰 데이터셋도 처리 가능
- 메모리 사용량 고정 (배치 크기에 비례)

**단점**:

- 디스크 I/O 병목 발생 가능
- 구현 복잡도 증가
- 데이터 증강 적용 복잡

**권장 사항**:

- 우선 해결책 2 (POI 사전 추출) 시도
- 여전히 메모리 부족 시 해결책 1 또는 3 적용

***

### Challenge 2: 학습 정체 (Training Plateau)

#### 원인 1: 잘못된 정규화

**증상**:

- 정확도가 20% 근처에서 정체
- Loss가 감소하지 않거나 매우 느리게 감소
- Training과 Validation 정확도가 모두 낮음

**진단**:

- 개별 트레이스 정규화(Horizontal) 사용 여부 확인
- 정규화 전후 데이터 분포 시각화

**해결**:

- Vertical Z-score 정규화로 즉시 전환
- 기대 효과: 정확도 20% → 60-70%로 즉시 상승


#### 원인 2: 클래스 불균형

**증상**:

- Validation loss는 감소하지만 accuracy는 정체
- Confusion matrix에서 대부분의 예측이 다수 클래스로 편중
- 소수 클래스(HW=0, 1, 22, 23)의 정확도 거의 0%

**진단**:

- 클래스별 샘플 수 분포 확인
- 클래스별 정확도(recall) 계산

**해결**:

- Class weights 적용
- 희소 클래스 필터링
- Focal loss 시도

**기대 효과**:

- 클래스별 정확도 균형 개선
- 전체 정확도 5-10% 향상


#### 원인 3: 학습률 문제

**증상 A: 학습률 너무 높음**

- Loss가 진동하며 수렴하지 않음
- Gradient exploding (NaN loss)

**증상 B: 학습률 너무 낮음**

- Loss 감소가 매우 느림
- 수십 에포크가 지나도 수렴하지 않음

**진단**:

- Learning curve 시각화
- Gradient norm 모니터링

**해결**:

- 초기 학습률 조정 (0.001 → 0.0005 또는 0.0001)
- ReduceLROnPlateau 적용하여 adaptive 조정
- Gradient clipping (극단적인 경우)


#### 원인 4: 과적합 (Overfitting)

**증상**:

- Training accuracy 95%, Validation accuracy 60%
- Training loss는 계속 감소하지만 Validation loss는 증가
- 큰 Train-Val gap

**진단**:

- Learning curve에서 Train/Val 곡선 발산 확인
- 모델 복잡도(파라미터 수) vs 데이터 크기 비율 계산

**해결**:

- Dropout 증가 (0.3 → 0.5)
- L2 regularization 추가 또는 강화
- POI 개수 감소 (500 → 300)
- 모델 단순화 (레이어 제거)
- Data augmentation (Gaussian noise)

**우선순위**:

1. Dropout 증가 (가장 쉽고 효과적)
2. POI 감소 (입력 차원 축소)
3. L2 regularization
4. 모델 단순화

***

### Challenge 3: 과적합 세부 대응

#### 진단 방법

**Learning Curve 분석**:

- X축: Epoch
- Y축: Loss 또는 Accuracy
- 두 개의 선: Training, Validation
- **과적합 패턴**: Train 성능은 계속 향상, Val 성능은 정체 또는 악화

**Train-Val Gap 계산**:

- Gap = Train Accuracy - Val Accuracy
- Gap > 10%: 과적합 의심
- Gap > 20%: 심각한 과적합, 즉시 대응 필요

**조기 탐지**:

- Validation loss가 3-5 에포크 연속 증가
- Training loss는 감소하는데 Validation loss는 증가


#### 해결책 우선순위

**1단계: Dropout 증가 (최우선)**

- 현재: Dropout(0.3 또는 0.4)
- 시도: Dropout(0.5 또는 0.6)
- 이유: 구현 간단, 효과 즉각적, 부작용 적음

**2단계: POI 개수 감소**

- 현재: 500 POIs
- 시도: 400, 300 POIs
- 이유: 입력 차원 감소 → 모델 복잡도 감소
- 효과: 파라미터 수 감소, 학습 속도 향상

**3단계: L2 Regularization 추가/강화**

- 현재: 없음 또는 1e-5
- 시도: 1e-4
- 적용: 모든 Dense, Conv Layer
- 효과: 큰 가중치 억제

**4단계: 모델 단순화**

- 레이어 개수 감소: 4층 → 3층
- 뉴런 수 감소: 512 → 256
- 효과: 모델 용량(capacity) 감소

**5단계: Data Augmentation**

- Gaussian Noise 추가 또는 강화
- stddev 0.05 → 0.07
- 효과: 가상의 데이터 증가, 일반화 향상


#### 실험 프로토콜

**단계적 접근**:

1. 한 번에 한 가지 변경만 적용
2. 각 변경 후 성능 평가
3. 개선되면 다음 단계, 악화되면 롤백
4. 모든 조합 테스트는 시간 소모 (Grid Search는 최후 수단)

**평가 지표**:

- Validation Accuracy (주지표)
- Train-Val Gap (과적합 정도)
- Guessing Entropy (최종 목표)

***

## 타 연구 대비 개선 포인트

### 우리의 강점

#### 1. 체계적 전처리 파이프라인

**우리의 접근**:

- Vertical Z-score 정규화 (필수)
- Data Power Trace 추출 (선택, 고급)
- Moving Average Filter (선택)
- 3단계 조합으로 최대 효과

**타 연구의 한계**:

- 많은 논문들이 정규화 방법을 명확히 기술하지 않음
- 또는 한 가지 방법만 사용

**우리의 차별점**:

- 각 전처리 단계의 효과를 독립적으로 측정
- 조합 효과 정량화
- 재현 가능한 파이프라인 제공


#### 2. 다층적 POI 선택 전략

**MLP용: Sparse POI**

- 목표: 전역적 정보 포착
- 방법: 상위 N개 직접 선택

**CNN용: Windowed POI**

- 목표: 시계열 패턴 보존
- 방법: 피크 중심 윈도우 확장

**타 연구**:

- 보통 한 가지 POI 전략만 사용
- 모델 특성에 맞춘 최적화 부족

**우리의 이점**:

- 모델별 최적 POI 선택
- 성능과 효율성의 균형


#### 3. 클래스 불균형 다각도 접근

**우리의 전략**:

- Class weights (필수)
- Label smoothing (보조)
- 희소 클래스 필터링 (데이터 정제)
- 클래스별 성능 분석

**타 연구**:

- 클래스 불균형 언급만 하고 해결 방법 부실
- 또는 한 가지 방법만 적용

**우리의 차별점**:

- 다층 방어 전략
- 정량적 효과 측정


### 타 연구에서 학습할 점

#### 1. Zaid's Efficient CNN 구조

**핵심 인사이트**:

- Kernel size 11 이상 사용의 중요성
- Stride=1 유지로 정보 손실 방지
- GlobalAveragePooling의 효과

**우리의 적용 계획**:

- Phase 1에서 정확히 재현
- 재현 성능이 논문과 일치하는지 검증
- 검증 후 우리 데이터에 맞게 튜닝

**기대 효과**:

- 검증된 아키텍처로 안정적 베이스라인
- 시행착오 최소화


#### 2. AutoPOI 프레임워크

**개념**:

- Deep Reinforcement Learning으로 POI 자동 선택
- 인간의 편향 없는 최적 조합 탐색

**현재 프로젝트에서의 위치**:

- 현재는 over-engineering (불필요)
- 장기 연구 과제로 고려

**활용 가능성**:

- 석사/박사 논문 주제
- 여러 데이터셋에 걸친 일반화 연구
- 새로운 암호 알고리즘 공격 시 빠른 적응


#### 3. Attention 메커니즘

**효과**:

- 모델이 자동으로 중요한 POI 집중
- 2-5% 성능 향상
- 해석 가능성 증대

**적용 계획**:

- Phase 2 또는 3에서 실험
- CBAM (Convolutional Block Attention Module) 추가
- Grad-CAM과 비교하여 Attention 일치도 분석

**주의사항**:

- 계산 비용 증가
- 하이퍼파라미터 추가 (reduction ratio 등)
- 성능 향상이 미미하면 생략


#### 4. Data Power Trace의 교차 바이트 효과

**타 연구 결과**:

- 단일 바이트 공격: 3.64배 효율 향상
- 교차 바이트 공격 (여러 계수 동시): 14.62배 효율 향상

**우리의 활용**:

- Phase 2에서 DPT 적용
- 단일 계수 공격으로 효과 검증
- 성공 시 여러 계수 동시 공격으로 확장

**연구 의의**:

- 실용적 공격 시간 대폭 단축
- 대응책 개발의 긴급성 강조

***

## 실행 로드맵

### Week 1-2: 데이터 탐색 및 전처리

**목표**: 데이터 완전 이해 및 전처리 파이프라인 구축

**세부 작업**:

- [ ] `profiling_40000_u=cs.npy` 로드 및 차원 구조 확인
    - 3D 배열의 각 차원 의미 파악
    - 특정 계수 선택 방법 결정
- [ ] Hamming Weight 변환 및 분포 분석
    - 클래스별 샘플 수 시각화 (히스토그램)
    - 희소 클래스 (<150 샘플) 식별
- [ ] `rho.npy` 로드 및 시각화
    - Pearson 상관계수 플롯
    - 높은 상관관계 구간 식별 (POI 후보)
- [ ] POI 추출 구현
    - Sparse POI (300, 400, 500개) 각각 추출
    - Windowed POI (700, 1000개) 추출
- [ ] Vertical Z-score 정규화 구현
    - 전체 Profiling 데이터의 통계 계산
    - Profiling 데이터 정규화
    - Attack 데이터를 Profiling 통계로 정규화
- [ ] Data Power Trace 구현 (선택)
    - Mean Trace 계산
    - 감산 적용
- [ ] Moving Average Filter 구현 (선택)
- [ ] 전처리된 데이터 저장 (캐싱)
    - POI별 `.npy` 파일 생성
    - 메타데이터 JSON 파일 생성

**산출물**:

- 전처리 스크립트: `preprocess.py`
- 경량 데이터셋: `profiling_poi400.npy`, `attack_poi400.npy`
- 전처리 보고서: `preprocessing_report.md` (데이터 분포, POI 선택 근거 등)
