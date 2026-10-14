# DilithiumSCA: Deep Learning Side-Channel Analysis of CRYSTALS-Dilithium

Research code for profiling-based **power analysis of CRYSTALS-Dilithium**, focusing on coefficient-wise Montgomery reduction leakage in the Number Theoretic Transform (NTT) domain. The v3 TensorFlow CNN predicts the Hamming weight of 32-bit intermediate values and evaluates calibrated candidate key rankings.

[English overview](#english-overview) · [한국어 안내](#한국어-안내)

## English overview

### What this repository provides

- A shared CNN trained on coefficient windows, with training progress logs and checkpoint saving.
- Hamming-weight classification, temperature calibration, and candidate-level guessing entropy (GE) / success rate (SR) evaluation.
- Saved experimental results, independent audit tools, and an analysis of related Dilithium side-channel attacks and public datasets.

The current implementation is in `project v3`; v1 and v2 are historical experiments. This is an experimental research repository, not a production cryptographic library or a verified full-key recovery benchmark.

### Recorded results and limitations

Training and inference finished on September 23, 2026. The saved results were audited on October 14, 2026.

- Attack-campaign Hamming-weight Top-1: **66.625%**, Top-3: **96.190%**, over 102,400 sampled coefficient windows. These are window draws, not 102,400 distinct power traces.
- Representative coefficient `(poly0, coeff50)`: **69.87%** Top-1 over 10,000 attack traces.
- For that coefficient, SR reached **100/100 trials from 8 traces** within a fixed sample of **65,254 candidate residues**. Exhaustive ranking over `q = 8,380,417` and simultaneous recovery of all 1,024 coefficients have **not** been verified.

Profiling set4 was reused for checkpoint selection and calibration. Some attack labels were used during earlier time-map development, so the current attack campaign is not a completely blind end-to-end test. The original dataset source and collection metadata remain unconfirmed. See the [results report](project%20v3/results_report.md) for metric definitions, audit coverage, and evaluation limitations.

### Getting started

```bash
git clone https://github.com/rolot789/DilithiumSCA.git
cd DilithiumSCA
```

1. Inspect the committed [evaluation results](artifacts/v3/evaluation_single/results.json), [audit record](artifacts/v3/evaluation_single/audit.json), and [training history](artifacts/v3/training_single/history.csv). Reading these records does not require the original traces.
2. Follow the [v3 environment and execution guide](project%20v3/README.md#7-사용법) before running the Python tools. The tested training environment used Python 3.11.15, NumPy 1.26.4, TensorFlow 2.16.2, and tensorflow-metal 1.2.0 on Apple Silicon. The root `requirements.txt` is not a locked v3 environment.
3. Training and inference require the original dataset and normalized arrays; inference also requires a trained checkpoint. The full audit requires the original dataset and the saved best/final checkpoints. These inputs are **not included** in a repository clone. The training and inference CLIs use existing normalized arrays; they do not automatically prepare the dataset.

After setting up the environment, these commands show the available options without starting training or inference:

```bash
./.venv-v3/bin/python "project v3/v3_train.py" --help
./.venv-v3/bin/python "project v3/v3_infer.py" --help
```

### Documentation

The detailed research notes are currently in Korean.

- [v3 implementation and execution guide](project%20v3/README.md)
- [Training, inference, and audit report](project%20v3/results_report.md)
- [Evidence-based training improvements](project%20v3/training_improvements.md)
- [Related work, comparable datasets, and research directions](project%20v3/related_work_analysis.md)
- [Initial v3 design and exploration notes](project%20v3/instruction_v3.md) — historical observations, distinct from the current CNN results.

## 한국어 안내

Dilithium의 계수별 Montgomery reduction 출력에 대한 전력 누설을 학습하고, Hamming Weight 예측과 후보 키 랭킹을 평가하는 연구 저장소다. 현재 실행 기준은 `project v3`이며 v1·v2는 이전 실험 기록이다.

## 현재 결과

2026년 9월 23일 v3 단일 HW 헤드 학습·추론을 완료했고, 2026년 10월 14일 저장된 결과를 재검산했다.

- 학습: 44/50 epoch에서 조기 종료, best epoch 36, validation Top-1 67.926%.
- 공격 캠페인: 102,400개 계수 윈도우 표본 Top-1 66.625%, Top-3 96.190%.
- 대표 계수 `(poly0, coeff50)`: 공격 트레이스 10,000개에서 Top-1 69.87%.
- 대표 계수의 65,254개 후보 표본에서 8트레이스 이후 SR 100% 관측(100회 실험). 전체 q 공간과 1024계수 전체 키 복구는 미검증.

## 읽을 문서

- [v3 README와 실행 안내](project%20v3/README.md): 프로젝트 설명, 환경과 재현 명령.
- [학습·추론 결과 보고서](project%20v3/results_report.md): 실측 수치, 지표 정의, 평가 한계, 재검증 결과.
- [다음 학습 개선 분석](project%20v3/training_improvements.md): 경계 계수 진단, 분할·보정·학습 실험 순서.
- [선행 연구 비교와 연구 개선안](project%20v3/related_work_analysis.md): 실제 측정 연구·공개 유사 데이터, 비교 조건과 후속 검증 설계. 로컬 데이터의 원출처는 아직 미확인.
- [v3 초기 설계와 탐색 기록](project%20v3/instruction_v3.md): 과거 실측과 합성 실험. 현재 CNN 결과는 결과 보고서를 기준으로 읽는다.

## 저장소 구성

```text
Dataset/1-Dataset/                 원본 profiling·attack 데이터 (로컬)
project v1/                       초기 실험
project v2/                       이전 CNN·평가 기록
project v3/                       현재 모델·학습·평가 코드와 문서
artifacts/v3/training_single/      best/final.keras, history.csv, config.json
artifacts/v3/evaluation_single/    results.json, target_predictions.npz, audit.json
artifacts/v3/*_norm.npy            정규화 memmap (로컬)
```

명령은 저장소 루트에서 실행한다. 현재 `.venv-v3`와 로컬 데이터·모델을 이용한 재검산은 아래와 같다.

```bash
./.venv-v3/bin/python "project v3/v3_audit_results.py"
```

새 학습과 평가 재실행 절차는 [v3 실행 안내](project%20v3/README.md#7-사용법)에 있다. 원본 데이터와 대형 모델·memmap은 로컬 산출물이므로 저장소만 복제해서 바로 실행할 수 있는 것은 아니다. 루트 `requirements.txt`는 이전 실험을 포함한 일반 의존성 목록이며, v3의 실제 검증 환경은 v3 README를 따른다.
