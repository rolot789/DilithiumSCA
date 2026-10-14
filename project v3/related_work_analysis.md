# Dilithium 부채널 선행 연구와 v3 개선 방향

## 기술 요약

현재 v3와 가장 가까운 비교 대상은 **NTT 영역의 Montgomery reduction 누설로 계수를 복구하고, SIS 기반 후처리로 비밀 벡터를 완성하는 연구**다. 다만 로컬 데이터의 원 배포 URL·DOI가 없어, 이번 조사에서 **동일 배포본을 사용한 연구는 확인하지 못했다**. 공개 DTDS는 같은 Dilithium 계열이지만 nonce `y`의 unpacking을 측정하는 다른 데이터다. [Qiao et al., When NTT Meets SIS](https://eprint.iacr.org/2023/1866), [Yuan et al., DTDS](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250048).

v3의 공격 HW 정확도 66.625%와 대표 계수의 8트레이스 이후 후보 표본 내 SR 100%는 유용한 개발 결과다. 그러나 선행 연구의 전체 `s1` 복구나 유효 서명 생성과 같은 종착점을 아직 검증하지 않았다. 다음 연구의 우선순위는 **같은 데이터의 기준선 비교 → 위치별 약점 개선 → 전체 후보 공간 평가 → 신뢰도 기반 부분 복구와 대수적 후처리 비교**다. 모델을 더 오래 학습하는 것만으로 이 차이가 해소되지는 않는다.

조사 기준일은 2026-10-14이다. 이 문서는 문헌 조사와 실험 설계 제안이며, 추가 학습·새 공격 실험·외부 대형 데이터 다운로드를 수행한 결과가 아니다. 우리 실측의 기준 문서는 [결과 보고서](results_report.md), 구현 중심 개선안은 [학습 개선 분석](training_improvements.md)이다.

## 1. 데이터 정체성과 비교 단위를 먼저 구분해야 한다

### 로컬 자료는 DTDS와 파일·공격 지점이 다르다

아래는 현재 파일 및 재검산 기록과 DTDS의 공개 설명을 대조한 것이다. 같은 `4×256` 라벨 구조만으로 같은 측정 캠페인이라고 판단할 수 없다.

| 구분 | 로컬 `Dataset/1-Dataset` | 공개 DTDS |
|---|---|---|
| 원출처 | 로컬 문서의 DSCAD 명칭 외 배포 URL·DOI 미확인 | 논문 DOI `10.11999/JEIT250048`, 데이터 DOI `10.57760/sciencedb.j00173.00001` |
| 측정·학습 대상 | 계수별 `u=MR(c*s)`; NTT pointwise 곱셈을 대상으로 해석 | `polyz_unpack`에서 생성되는 nonce `y` |
| 트레이스 | profiling 40,000개 + attack 10,000개 | 총 60,000개; 논문 설명은 profiling 50,000개 + attack 10,000개 |
| 파일별 파형 | 10,000×40,000, int64 | `polyz_unpack_traces.npy`: 5,000×20,000 |
| 중간값 | `profiling_40000_u=cs.npy`, `attack_10000_u=cs.npy` | `polyz_unpack_y.npy`: 5,000×4×256 |
| 부가 정보 | attack `c`, `s`; 현재 사용 자료에서 공개키·서명·retry 연결 정보 미확인 | `mate.req`에 키·메시지·서명 등 메타데이터 |
| 수집 장치·펌웨어 | 현재 자료만으로 장치, ADC, compiler, commit 확정 불가 | STM32F405RGTx / ChipWhisperer UFO; 25 MHz 실행, 125 MHz 샘플링; GCC 10.2.1, `-O1` |

DTDS 측 정보는 [논문 표 2·3 및 데이터 설명](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250048?viewType=HTML), [공식 데이터 DOI](https://doi.org/10.57760/sciencedb.j00173.00001)에 근거한다. 로컬 측은 [재검산 산출물](../artifacts/v3/evaluation_single/audit.json)과 기존 결과 보고서에 근거한다. 로컬 `u=MR(c*s)` 관계는 저장된 원래 정수 대표원으로 10,240,000개 값 모두 일치했지만, 이것만으로 수집 펌웨어나 공개 배포본의 신원을 확정할 수는 없다.

원출처 확인에는 원 다운로드 URL, 배포 README, 수집 코드/바이너리, 해시, NTT 변환 규약, 프로파일링 키 생성 방법, 장치·세션 식별자가 필요하다. 확인 전에는 로컬 자료를 특정 논문의 공식 데이터라고 소개하지 않는다. 변환·부분 발췌된 배포본일 가능성도 원본 메타데이터 없이는 판별할 수 없다.

### 정확도·계수 복구·전체 키 복구는 서로 다른 결과다

- **HW Top-1**: 한 윈도우의 32비트 2의 보수 HW를 맞힌 비율. 현재 v3는 클래스 0~32의 33클래스 모델이다.
- **계수 SR**: 선택한 계수의 정답이 유일한 최고 점수 후보가 된 반복 실험 비율. 후보 표본인지 전체 `q`인지 반드시 함께 적는다.
- **전체 벡터 SR**: 같은 트레이스 집합에서 목표 비밀 벡터 전체를 정확히 복구한 반복 비율. 개별 계수 SR의 평균으로 대체하지 않는다.
- **종단간 공격 성공**: 복구한 값으로 구현의 공개 관계를 검증하거나 실제 유효 서명을 생성하는 단계까지 통과한 결과. 전체 원래 secret-key byte열 복원과 동등 키 복구도 구분한다.
- **트레이스 수**: 모델 학습용 profiling 측정 수와 공격 시 관측 수를 별도로 기록한다. 한 측정에서 나온 1024개 윈도우는 1024회의 독립 측정이 아니다.

우리의 102,400개 공격 표본은 반복 추출한 계수 윈도우 수다. 8트레이스 결과는 `(poly0,k50)` 한 계수의 65,254개 후보 표본, 100회 실험이다. 전체 후보 `q=8,380,417` 및 1024계수 동시 복구는 미검증이다. 상세 GE·SR·동점 정의는 [현재 결과 보고서](results_report.md)에 있다.

## 2. 실제 측정 기반 연구는 모델 예측 이후의 복구까지 연결한다

아래 표는 연구별 결과와 그 결과가 성립하는 조건을 함께 읽기 위한 비교표다. 동일 데이터·동일 실험의 순위표가 아니며, 표의 정확도·성공률은 같은 분모가 아니다. ePrint 번호는 최초 공개 연도를 나타내며, NIST 발표본과 이후 개정판의 수치를 섞지 않았다.

| 연구 | 누설·장치 / 방법 | 원문이 보고한 결과 | v3와의 핵심 차이 |
|---|---|---|---|
| [Qiao et al., 2023/1866](https://eprint.iacr.org/2023/1866), 2024-07 개정 | Cortex-M4, NTT Montgomery 출력; CPA + SIS/BKZ 후처리 | 5,000 power traces로 Dilithium2의 전체 `s1` 복구, 2.4분 | 공격 지점이 가장 가깝지만, 우리 데이터와 동일 배포본 여부 미확인; 후처리와 전체 복구가 포함됨 |
| [Qiao et al., 2024/512](https://eprint.iacr.org/2024/512), 2024-04 개정 | STM32F4; 정상 다항식 영역 `z=y+cs1`; LR/CNN + 오류 포함 방정식 처리 | 한 서명에서 비밀키 복구 확률 60%, 1시간 이내; 두 서명에서 5분 이내 공격 | NTT 단일 계수 HW가 아니라 정상 영역 값과 공개 서명 제약을 함께 이용 |
| [Wang et al., NIST 2024 발표본](https://csrc.nist.gov/csrc/media/Events/2024/fifth-pqc-standardization-conference/documents/papers/single-trace-side-channel-attacks.pdf) | Cortex-M4, 비밀키 unpacking; 위치 유형별 MLP + 선형대수/격자 후처리 | `t0`가 알려진 변형의 전체 `s1` SR: 1회 9%, 10회 83%, 100회 99%, 1000회 100% | 추가 정보 `t0`, 다른 누설 지점, profiling과 다른 공격 장치; 알려지지 않은 `t0` 변형은 별도 조건 |
| [Marzougui et al., 2022/106](https://eprint.iacr.org/2022/106) | Cortex-M4, nonce bit-unpacking; ML + least squares/ILP | 실제 잡음 측정에서 동등 키 복구, 임의 메시지 서명 가능 | 이진 누설 판단을 구조적 제약에 연결; 일반 HW 정확도와 직접 비교 불가 |
| [Liu et al., 2025/582](https://eprint.iacr.org/2025/582), 2025-12 개정 | Cortex-M4 reference·ASM 구현; accepted/rejected 누설의 통합 factor graph/BP | rejected 누설 포함 시 전체 키 복구용 트레이스 최소 50% 감소; 자체 조건에서 rejected만으로 30회 미만 | 서명 생성 재시도·거절 정보와 여러 누설의 관계가 필요; 현재 `u,c,s`만으로 재현 불가 |

수치는 각 원문이 주장한 결과이며 우리 환경에서 재현하지 않았다. 공격 시간은 각 논문 설정의 계산 시간이며, profiling·수집 비용 포함 여부가 같다고 가정하지 않는다. 특히 `8 < 30 < 5000`이라는 숫자 순서로 공격 성능의 우열을 정할 수 없다.

### 2.1 NTT→SIS: 약한 계수를 모두 따로 해결하지 않아도 되는 방향

Qiao 등의 연구는 NTT의 구조를 SIS 문제에 연결하고 분할·차원 축소로 후처리 비용을 낮춘다. 원문 초록은 특정 BKZ-60 구성에서 필요한 계수 수가 65→32로 줄었다고 보고한다. 이를 “우리의 1024계수 중 아무 32개면 충분하다”로 해석하면 안 된다. 축소 문제의 차원·계수 선택 조건과 구현 규약을 확인해야 한다. [원문 및 개정 이력](https://eprint.iacr.org/2023/1866).

우리에게 적합한 가설은 **보정된 후보 점수로 신뢰도 높은 NTT 계수부터 복구하고, 작은 정상 영역 secret의 구조로 남은 계수를 구하는 방식**이다. HW 예측의 argmax가 곧 NTT 계수 값은 아니므로 먼저 후보 랭킹이 필요하다. 계수 confidence만 보고 잘못된 값을 확정하면 후처리가 실패할 수 있어, 복수 후보·오류 허용·성공/실패 조건도 설계해야 한다.

현재 `s`가 실제 Dilithium의 작은 secret을 어떤 NTT·Montgomery 표현으로 변환한 값인지, 정상 영역에서 필요한 계수 범위가 성립하는지 확정되지 않았다. 이를 확인한 뒤에만 같은 조건의 SIS 실험으로 연결한다. 필요한 변환 규약이 없으면 공개 데이터나 새 수집으로 검증 범위를 넓혀야 한다.

### 2.2 정상 영역 값과 공개 관계를 쓰는 연구는 HW만 예측하지 않는다

Qiao 등의 2024 연구는 `y`, 정상 영역 `cs1`, 공개 `z`의 관계와 오류가 있는 방정식을 활용한다. CNN을 200개 서명의 누설로 학습한 조건에서 `cs1` 복구율 74%, 결합 누설 사용 시 92%를 보고한다. 이는 우리 33클래스 HW Top-1 66.625%와 다른 지표다. 한 서명에서 다수 계수 윈도우를 얻는 학습 확장도 원문에 나타난다. [초록](https://eprint.iacr.org/2024/512), [원문 §5의 CNN 학습 설명](https://eprint.iacr.org/2024/512.pdf).

따라서 1024계수 윈도우를 풀링한다는 아이디어 자체를 새 기여로 주장하기 어렵다. 개선 실험은 풀링의 유무뿐 아니라 위치 조건부 모델, 약한 계수의 후보 축소, 보정된 posterior를 후처리에 전달했을 때의 종단간 이득을 확인해야 한다. 현재 데이터에는 `z,y`가 없어 이 논문의 결합 공격을 그대로 적용할 수 없다.

### 2.3 위치 유형별 모델과 다른 장치 평가는 유용한 대조군이다

Wang 등의 NIST 발표본은 440샘플 입력의 5클래스 모델을 사용하며, unpacking 위치의 8가지 유형을 구분한다. compiler 처리 차이로 `j mod 8 = 7` 유형의 누설이 약해지는 문제도 설명한다. 전체 복구 수치는 표 5의 `t0` 알려짐 조건이고, 표 6의 격자 변형과 구별해야 한다. [NIST 원문 §6–7, 표 5·6](https://csrc.nist.gov/csrc/media/Events/2024/fifth-pqc-standardization-conference/documents/papers/single-trace-side-channel-attacks.pdf).

우리의 경계 8계수 Top-1 평균은 50.119%, 내부 표본 29계수는 67.171%로 17.052%p 차이가 난다. 이것은 위치를 완전히 동일시하는 모델의 한계를 시험할 근거다. 하지만 논문의 unpacking 주기 8을 우리 Montgomery 연산에도 고정해서 적용할 근거는 없다. 먼저 자체 파형·가능하면 disassembly에서 경계, poly, 반복 주기의 차이를 확인한다.

위치별 독립 모델 1024개보다 **공유 trunk + 작은 위치 조건부 head**를 우선 대조군으로 제안한다. 모델 크기·총 업데이트·추론 비용을 함께 측정하고, 새로운 공격 장치로 전이가 가능한지는 별도의 측정 없이는 주장하지 않는다.

### 2.4 이진 누설도 종단간 검증에 연결될 수 있다

Marzougui 등의 연구는 작은 unpacking 누설을 학습한 뒤 least squares와 ILP로 서명 가능한 동등 키를 얻는다. [ePrint 원문](https://eprint.iacr.org/2022/106), [저자 공개 코드](https://github.com/fgsect/profiling-attack-dilithium). 정답 클래스 수나 평균 정확도보다 얻은 정보가 암호 구조를 통해 어떻게 합쳐지는지가 중요하다는 비교 사례다.

우리도 HW·sign·byte-HW 헤드를 추가할 때 각각의 정확도 상승만 측정하지 말고, 기존 HW 모델 대비 실제 후보 점수 및 전체 복구 성공의 한계 기여를 측정해야 한다. 서로 종속적인 헤드를 독립 확률처럼 곱하거나 PI를 단순 합산하면 같은 정보를 중복 계산할 수 있다.

### 2.5 rejected signature 활용은 유망하지만 현재 데이터의 즉시 개선은 아니다

Liu 등의 연구는 유효 서명뿐 아니라 거절된 서명 생성 시도의 누설을 BP로 통합한다. 공개 코드도 제공되지만 저장소는 데이터 버전 변경을 명시한다. [2025-12 개정 초록](https://eprint.iacr.org/2025/582), [저자 구현](https://github.com/AIGIUS/BP-On-ML-DSA).

현재 고립된 pointwise 측정은 accepted/rejected 여부, 같은 서명 생성에 속한 retry 그룹, 대응 공개 출력이 확인되지 않는다. 그러므로 이 방식은 기존 데이터를 증강해 바로 얻는 성능 향상이 아니라 **새 수집 설계의 후보**다. 기존 결과와는 별도 실험군으로 관리하고, signed attempt·accepted signature·power trace의 개수를 각각 기록한다.

## 3. 공개 유사 데이터는 검증할 질문에 맞춰 선택해야 한다

외부 데이터의 우선순위는 이름의 유사성보다 검증하려는 가설에 따라 달라진다. 다음 자료들은 동일 데이터 복제본으로 확인된 것이 아니라 독립적인 후속 검증 후보다.

| 자료 | 검증하기 좋은 질문 | 그대로 가져올 수 없는 것 |
|---|---|---|
| [DTDS, 2025](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250048) / [데이터 DOI](https://doi.org/10.57760/sciencedb.j00173.00001) | Dilithium의 다른 누설 지점에서도 위치 조건부·보정·풀링이 유효한가? 메타데이터 기반 종단간 평가가 가능한가? | Montgomery 라벨, 현재 time-map, 학습 가중치의 직접 유효성 |
| [Kyber pair-pointwise 데이터, 2025/811](https://eprint.iacr.org/2025/811) | NTT 곱셈 계열에서도 공유 표현·후보 랭킹의 평가 프로토콜이 작동하는가? | Kyber의 pair-pointwise 식과 파라미터는 다름; Dilithium 후보 점수 식·HW 라벨을 그대로 재사용 불가 |
| [Two Birds, One Mask: 공개 코드·데이터, 2026, v2 고정 링크](https://zenodo.org/records/18702355) | 비보호/마스킹된 ML-DSA·ML-KEM 하드웨어의 누설·방어 평가를 어떻게 확장할 것인가? | 소프트웨어 모델의 직접 전이, fixed/random TVLA 입력이 곧 profiling/attack 라벨이라는 가정 |
| [ANSSI ASCAD 공식 자료](https://github.com/ANSSI-FR/ASCAD) | 고정 split, MLP/CNN 기준선, 후보 rank 평가의 재현성 구조를 어떻게 마련할 것인가? | AES의 256개 byte 후보와 정확도를 Dilithium의 q공간 성능과 직접 비교하는 것 |

DTDS 논문은 자체 deep-learning 평가에서 모델링 수 증가에 따라 표 7의 정확도가 9.98%→30.29%로 증가했다고 보고한다. 최종 점의 모델링 수는 40,000이고 POI 수는 500이다. 그러나 확보한 자료만으로 그 정확도의 정확한 라벨 변환·평가 분모를 우리 HW 지표와 동등하게 정렬하지 못했다. **“v3 66.625%가 DTDS 30.29%보다 우수하다”는 결론은 내리지 않는다.** [저널 원문 표 7 및 §4.3](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250048?viewType=HTML).

Kyber 자료는 PQClean reference decapsulation의 NTT pair-pointwise 측정을 공개한 데이터셋 논문이다. 공개 초록은 특정 공격 성공 수치를 제시하지 않으므로 성능 기준선을 임의로 부여하지 않는다. [원문](https://eprint.iacr.org/2025/811). 하드웨어 자료는 v2를 명시한 링크이며, 후속 사용 시 최신 버전·라이선스·파일별 라벨을 다시 확인해야 한다. 이번에는 외부 파형을 다운로드하거나 모델을 재학습하지 않았다.

## 4. 우리 연구의 기여는 아직 ‘검증할 후보’ 단계다

현재 실측에서 말할 수 있는 것은 전체 위치를 공유하는 CNN이 이 로컬 측정의 HW 정보를 학습했고, 확률 보정이 NLL/ECE를 개선했으며, 대표 계수의 작은 후보 표본에서 소수 트레이스로 높은 SR을 관측했다는 것이다. 아래 구분은 관련 연구 대비 주장 범위를 정리한다.

| 항목 | 현재 판단 | 연구 결과로 만들기 위한 검증 |
|---|---|---|
| 한 트레이스에서 다수 계수 윈도우 확보 | 선행 사례가 있어 단독 새로움 부족 | 동일 조건의 per-position·pooled·conditional 비교 |
| HW 단일 CNN의 높은 평균 정확도 | 로컬 개발 결과; 기존 연구 대비 우월성 미확인 | 동일 데이터 CPA/템플릿/MLP, 같은 라벨·split·후보 식 |
| 경계 계수 및 위치 조건부 모델 | 관측된 병목에 맞는 차별화 후보 | 1024계수 전량, 경계/내부 분리, 약한 계수 rank 이득 |
| 확률 보정·prior와 공격 성능의 관계 | NLL/ECE 개선은 관측; 전체 복구 이득 미확인 | 분리 calibration, 동일 트레이스의 paired GE/SR 비교 |
| 부분 NTT 복구 + 구조적 후처리 | 가까운 선행 연구가 있음 | 정확한 표현·조건 검증, 현재 CNN posterior의 추가 기여 |
| 8트레이스 전체 키 복구 | 현재 근거 없음 | q공간 및 모든 목표 계수/비밀 벡터의 종단간 검증 |

가능한 연구 질문은 다음처럼 좁히는 것이 좋다: **“고정된 profiling 예산에서 위치 조건부·확률 보정 모델이 평균 HW 정확도보다 약한 NTT 계수의 후보 축소와 종단간 복구를 더 개선하는가?”** 답을 확인하기 전에는 새로운 SOTA나 세계 최초로 표현하지 않는다.

## 5. 다음 실험은 기준선과 약한 위치를 먼저 비교한다

### A. 지금 데이터로 가능한 비교 프로토콜을 고정한다

모든 실험에 트레이스 단위 split, train-only 시간 매핑, 전처리 통계·설정, 모델 선택·calibration 분리, 후보 대표원, 후보/트레이스 seed를 고정한다. 현재 attack 일부 라벨을 초기 매핑에 사용했으므로 기존 attack은 개발 평가로 남긴다. 최종 blind 성능은 설계에 쓰지 않은 새 캠페인에서 검증해야 한다.

기준선은 **CPA, 공분산을 정칙화한 Gaussian HW template, MLP, 기존 v3 CNN**으로 구성한다. 템플릿은 희소 HW 클래스의 공분산 추정에 shrinkage/대각 대조군을 둔다. CPA의 POI 선택은 profiling에서만 하고, CPA에는 신경망과 같은 가중치 학습은 없지만 POI 선택·후보 점수 계산 비용이 있다는 점을 표기한다. 모든 모델은 실제 구현을 따르는 MR 후보 식을 사용하지만, CPA의 correlation 점수와 CNN/template의 log-likelihood 점수를 억지로 같은 확률이라 부르지 않는다.

학습 모델 간에는 총 optimizer update와 윈도우 추출 횟수를 맞추는 비교와, wall-clock·메모리·추론 처리량 비교를 모두 기록한다. CPA·템플릿에 동일 epoch 수를 적용하는 것은 공정성 조건이 아니다. 유망한 모델은 최소 3개 학습 seed로 변동을 보고하고, attack 반복은 모델마다 같은 트레이스 순서를 사용한다. 이 수치는 실험 계획의 출발점이지 최적 설정이라는 주장이 아니다.

### B. 공유 모델과 위치 조건부 모델을 한 번에 비교한다

다음 ablation은 한 요소씩 바꾸고, profiling 진단으로만 구조를 결정한다.

1. 기존 96샘플 공유 CNN을 대조군으로 보존한다.
2. shared trunk에 poly 식별자와 경계 여부를 조건으로 제공하는 작은 head를 비교한다.
3. 자체 파형에서 반복 주기가 확인되는 경우에만 그 주기별 head를 추가한다.
4. 경계/내부의 offset·짧은 중심/긴 문맥 branch를 비교한다.
5. 약한 위치 oversampling을 추가하되 calibration prior와 위치별 지표를 재측정한다.

전체 1024계수를 평가하고 평균과 함께 계수별 NLL의 p95·최댓값, PI의 p5·최솟값, 경계/내부 Top-1, 계수 SR을 기록한다. 현재 경계 격차가 문맥 차이인지 매핑 오차인지 확정되지 않았으므로, 조건부 head의 성능 개선도 원인 규명과 구분해 해석한다.

### C. 확률 품질을 후보 랭킹과 연결한다

별도 calibration에서 온도, empirical HW prior, 보정 전/후 예측 평균 prior를 비교한다. 공격 NLL이 현재 0.865842→0.838323으로 줄었다는 관측은 좋은 출발점이지만, 같은 조건의 GE/SR 이득이 있는지는 아직 확인해야 한다. 멀티태스크의 sign·byte-HW 확률 결합은 종속성을 고려하고 실제 rank로 채택 여부를 정한다.

SR 실험의 트레이스 풀·순서·반복 수·최대 N을 맞춰 CNN, 기준선, oracle을 paired 비교한다. 반복이 같은 트레이스 풀을 공유하므로 102,400개 윈도우를 독립 표본으로 보는 신뢰구간은 사용하지 않는다. trace/session 단위 재표집과 학습 seed 변동을 별도로 보고한다. 100/100 성공은 해당 반복에서의 관측이지 일반적인 실패 확률 0의 증명이 아니다.

### D. 전체 후보 공간과 두 가지 복구 전략을 검증한다

첫 단계는 대표 계수와 약한 계수의 `q=8,380,417` 후보 점수를 블록별로 전량 계산하는 것이다. 메모리 제한 때문에 후보 수를 임의로 줄일 필요는 없으며, 전량 계산의 처리 시간·블록 크기·overflow 동작을 기록한다. 정수 대표원 변경으로 MR 라벨이 0.061416% 달라진 기존 재검산 결과를 고려해 저장 표현과 후보 표현을 먼저 맞춘다.

그다음 두 전략을 동일 입력에서 비교한다.

- **독립 복구**: 공통 N개 트레이스로 모든 목표 NTT 계수를 개별 복구하고, 1024계수 동시 성공률을 측정한다.
- **부분 복구 + 후처리**: calibration에서 정한 신뢰도 규칙으로 계수/후보를 선택하고, 검증된 NTT 변환과 작은 secret 구조를 이용해 남은 값을 복구한다. 정답을 보고 쉬운 계수를 고르지 않는다.

후처리에는 선택 계수 수, 선택 오류율, solver 시간·메모리·실패율을 기록한다. 전체 NTT 복구와 정상 영역 `s1` 복구를 구별한다. 공개키·서명이 없는 기존 자료에서는 원래 라벨 일치까지 검증할 수 있지만, 실제 유효 서명 생성까지 입증하려면 추가 메타데이터가 필요하다.

### 실행 가능성에 따른 우선순위

아래 표의 “가능”은 연구 설계를 의미하며 이번에 실행했다는 뜻이 아니다.

| 우선순위 | 작업 | 기존 데이터로 가능? | 완료 판정 |
|---|---|---|---|
| P0 | 원출처, 표현 규약, split·전처리 manifest 확정 | 일부 가능; 배포/펌웨어 정보 필요 | provenance 공백과 공격 식의 적용 범위 명시 |
| P1 | 동일 데이터 CPA/템플릿/MLP/CNN 기준선 | 가능 | 동일 평가의 계수 rank·비용 비교표 |
| P1 | 1024계수 진단, 경계·위치 조건부 ablation | 가능 | 약한 위치 개선과 내부 성능 손익, seed 변동 |
| P2 | calibration/prior/멀티태스크 rank 비교 | 가능 | held-out 확률 품질 + paired GE/SR |
| P2 | q공간 전량 및 1024계수 동시 성공 검증 | 가능하나 평가기 확장 필요 | 표본 근사 없는 순위와 동시 SR |
| P3 | 부분 NTT 복구 + SIS 후처리 | 조건부: secret 구조·NTT 규약 확인 필요 | 전체 `s1` 정확 복구 및 solver 실패/비용 |
| P3 | 유효 서명 생성으로 종단간 검증 | 공개키·서명/키 메타데이터 추가 필요 | 새 메시지의 유효 서명 검증 |
| P4 | DTDS·Kyber 외부 재현 / 다른 장치 blind 평가 | 외부 자료·독립 학습 또는 새 수집 필요 | 데이터별 독립 프로토콜과 일반화 한계 |
| P4 | rejected 누설·마스킹 하드웨어 | 새 측정 범위와 라벨 설계 필요 | 추가 정보/방어 조건을 명시한 별도 결과 |

## 6. 조사·검증 범위와 남은 질문

### 이번 비교는 제한 조건을 붙여 공유할 수 있다

검색은 로컬 파일명·DSCAD 명칭, Dilithium 데이터셋, Montgomery/NTT 공격, profiling 및 full-key recovery를 중심으로 수행했다. 수치 출처는 저자 ePrint, NIST 원문, 저널, 저자 코드/공식 데이터 저장소로 제한했다. 검색 결과에 뜬 2차 요약의 공격 수치를 다른 논문의 확정 결과처럼 가져오지 않았다. 전 분야의 완전한 체계적 문헌고찰이나 최신 SOTA 순위표는 아니다.

확보 수준은 다음과 같다. 동일 문서의 공개 초록과 원문 세부 조건을 구별했으며, 접근 제한 때문에 재현을 위해 추가 확인할 항목도 남겼다.

| 자료 | 확인한 근거 | 이번에 확인하지 않은 것 |
|---|---|---|
| 2023/1866 | 공식 초록·개정일, 원문 검색 추출 | 모든 표/solver 조건의 독립 재현, 로컬 데이터와 동일성 |
| 2024/512 | 공식 초록·개정일, 원문 실험·풀링 설명의 검색 추출 | 전체 PDF 연속 열람 및 코드 재현 |
| Wang NIST 발표본 | 원문 실험 조건·위치 모델·표 5·6, 공식 ePrint 초록 | 실제 파형/학습/공격 재현 |
| 2022/106 | 공식 초록, 저자 구현 README | 최신 출판본과 ePrint 수치의 전부 대조, 공격 재현 |
| 2025/582 | 2025-12 개정 초록, 저자 공개 저장소 | 전체 factor graph·각 표의 세부 조건, 재현 |
| DTDS | 저널 서지·초록, 표 2·3·7의 공개 검색 추출, 데이터 DOI | 원시 파일, 정확한 라벨 변환/분모, 로컬과의 해시 대조 |
| Kyber / Two Birds / ASCAD | 저자 데이터 설명 / 버전 고정 데이터 기록 / 공식 저장소 | 외부 파형 검증·재학습·우리 환경의 성능 |

일부 ePrint PDF와 저널 본문은 직접 접근이 제한돼 공개 초록·공식 사이트의 검색 추출을 사용했다. 그런 경우 미확인 내용을 확정적인 실험 조건으로 채우지 않았다. 논문 수치는 저자 보고값이며, 우리 수치는 저장 결과와 기존 재검산 범위에 한정한다. 외부 간 성공률을 하나의 그래프로 순위화하지 않은 이유는 공격 대상·분모·추가 정보가 다르기 때문이다.

후속 의사결정에 필요한 질문은 다음과 같다.

1. 로컬 데이터의 원 URL·DOI·수집 코드와 실제 MCU/펌웨어는 무엇인가?
2. `s`는 실제 작은 secret의 NTT인가, 임의 큰 정수의 곱셈 실험인가? NTT 및 Montgomery 스케일·순서는 무엇인가?
3. profiling과 attack의 키·장치·세션은 어떻게 달라지는가? 같은 계수 윈도우를 다른 split에 넣지 않았는가?
4. 최종 목표는 모든 NTT 계수, 정상 영역 `s1`, 동등 서명 키 중 무엇인가?
5. 공개키·실제 서명·독립 캠페인을 확보해 최종 종단간 검증을 할 수 있는가?

## 참고문헌과 재현 출발점

- Zehua Qiao, Yuejun Liu, Yongbin Zhou, Mingyao Shao, Shuo Sun. [When NTT Meets SIS: Efficient Side-channel Attacks on Dilithium and Kyber](https://eprint.iacr.org/2023/1866). ePrint 2023/1866; 조사 시 개정일 2024-07-01.
- Zehua Qiao, Yuejun Liu, Yongbin Zhou, Yuhan Zhao, Shuyi Chen. [Single Trace is All It Takes: Efficient Side-channel Attack on Dilithium](https://eprint.iacr.org/2024/512). ePrint 2024/512; 조사 시 개정일 2024-04-14.
- Ruize Wang, Kalle Ngo, Joel Gärtner, Elena Dubrova. [Single-Trace Side-Channel Attacks on CRYSTALS-Dilithium: Myth or Reality?](https://eprint.iacr.org/2023/1931). ePrint 2023/1931; 수치 비교에는 [2024 NIST 발표본](https://csrc.nist.gov/csrc/media/Events/2024/fifth-pqc-standardization-conference/documents/papers/single-trace-side-channel-attacks.pdf)을 사용.
- Soundes Marzougui, Vincent Ulitzsch, Mehdi Tibouchi, Jean-Pierre Seifert. [Profiling Side-Channel Attacks on Dilithium: A Small Bit-Fiddling Leak Breaks It All](https://eprint.iacr.org/2022/106). ePrint 2022/106; [저자 코드](https://github.com/fgsect/profiling-attack-dilithium).
- Zheng Liu, An Wang, Congming Wei, Yaoling Ding, Jingqi Zhang, Annyu Liu, Liehuang Zhu. [Release the Power of Rejected Signatures: An Efficient Side-Channel Attack on the ML-DSA Cryptosystem](https://eprint.iacr.org/2025/582). ePrint 2025/582; 조사 시 개정일 2025-12-15; [저자 코드](https://github.com/AIGIUS/BP-On-ML-DSA).
- Qingjun Yuan, Haojin Zhang, Haopeng Fan, Yang Gao, Yongjuan Wang. [DTDS: Dilithium Dataset for Power Analysis](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250048). Journal of Electronics & Information Technology, 2025, 47(8):2499–2508. [데이터 DOI](https://doi.org/10.57760/sciencedb.j00173.00001).
- Azade Rezaeezade, Trevor Yap, Dirmanto Jap, Shivam Bhasin, Stjepan Picek. [Side-Channel Power Trace Dataset for Kyber Pair-Pointwise Multiplication on Cortex-M4](https://eprint.iacr.org/2025/811). ePrint 2025/811; 조사 시 개정일 2025-06-04.
- Patrik Dobias, Aikata, Azade Rezaeezade, Lejla Batina, Lukas Malina, Tomas Gerlich. [Two Birds, One Mask: Side-Channel-Resistant Unified Hardware for ML-DSA and ML-KEM — 코드·데이터 기록](https://zenodo.org/records/18702355). 2026, v2 고정 링크; 후속 실험은 사용할 버전을 다시 확인하고 고정.
- ANSSI. [ASCAD 공식 데이터·모델·평가 코드](https://github.com/ANSSI-FR/ASCAD). 방법론 참고이며 Dilithium 수치 비교 대상은 아님.

결론적으로 지금 필요한 것은 논문의 낮은 트레이스 수를 목표 숫자로 가져오는 것이 아니라, **같은 데이터에서 공정한 기준선을 만들고 모델의 확률을 검증된 전체 복구 과정에 연결하는 것**이다. 이 문서의 제안은 후속 실험의 판단 기준이며, 아직 얻지 않은 개선 성능을 약속하는 결과가 아니다.
