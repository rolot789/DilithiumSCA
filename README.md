# DilithiumSCA

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
