---
name: dilithium-v3-run
description: Dilithium v3 부채널 공격 모델을 처음부터 끝까지 실행하고 리포트를 생성한다. 환경 점검, 데이터 검증, 전처리, 시간 매핑 보정, 학습, 3단계 평가, 마크다운 리포트까지 단계별로 진행하며 각 단계마다 통과 조건을 확인한다. 사용자가 "v3 학습 돌려줘", "모델 학습하고 리포트 뽑아줘", "이식성 측정해줘", "GE 재줘" 같은 요청을 할 때 사용한다. Apple Silicon(M시리즈) 환경을 기본 가정한다.
---

# Dilithium v3 실행 및 리포트 생성

`project v3/`의 모듈을 순서대로 실행해 학습과 평가를 완주하고 리포트를 만든다.

## 핵심 원칙

1. **각 단계는 게이트다.** 통과 조건을 만족하지 못하면 다음 단계로 넘어가지 말고
   사용자에게 보고한다. v2의 실패는 전부 "조용히 통과"에서 나왔다.
2. **정확도는 항상 다수 클래스 베이스라인(8.68%)과 함께 보고한다.**
   무작위 3.03%와 비교하면 성능이 과장된다.
3. **공격셋으로 하이퍼파라미터를 고르지 않는다.** 온도와 사전확률은 반드시
   캘리브레이션셋에서만 학습한다.
4. **수치를 지어내지 않는다.** 실행하지 못한 항목은 "미실행"으로 남긴다.

## 사전 준비

```python
import sys, os
sys.path.insert(0, "project v3")
DATASET = "../Dataset"     # 원본 DSCAD 파일 위치
WORK    = "v3_work"        # 중간 산출물 디렉터리
os.makedirs(WORK, exist_ok=True)
```

---

## Step 0. 환경 점검

```python
import v3_backend as bk
bk.configure_numpy_threads()
info = bk.configure_tensorflow()
```

**확인할 것**
- `apple_silicon: True`
- `gpu_devices`가 비어있지 않을 것 (비었으면 `pip install tensorflow-metal`)

**통과 조건**: Metal GPU 인식. 인식 안 되면 CPU로도 돌아가지만 매우 느리므로
사용자에게 알리고 계속할지 확인한다.

---

## Step 1. 데이터 검증 게이트

```python
import v3_common as vc
report = vc.verify_dataset(DATASET)
```

**확인할 것**
- 모든 파일의 shape/dtype이 기대값과 일치
- `mr_relation_ok: True` — `u = MR(c*s)` 관계 성립
- `s_plus_q_invariant: True` — 후보 공간이 q개임을 보증
- `majority_class_accuracy`가 약 0.0868

**통과 조건**: 예외 없이 완료. 예외가 나면 파일 조합이 잘못된 것이므로
**절대 진행하지 말고** 사용자에게 보고한다.

---

## Step 2. 시간 매핑

계수 `(j,k)`의 누설이 몇 번째 시간 샘플에서 일어나는지를 담은 표다.
**이 값이 틀리면 학습 자체가 성립하지 않는다.**

기존 실측 테이블을 쓰는 것이 기본이다.

```python
centers = vc.load_time_map("project v3/time_map.json")
```

새로 측정해야 할 경우(장비나 데이터셋이 바뀐 경우):

```python
import numpy as np
traces = np.asarray(vc.load_array(DATASET, "attack_10000traces.npy")[:600])
u_atk  = vc.load_array(DATASET, "attack_10000_u=cs.npy")
anchors = [0, 32, 64, 96, 128, 160, 192, 224, 255]
# 먼저 앵커 피크를 찾고(calibrate_time_map), 그 결과를 dense에 넘긴다
anchor_peaks = {}
for j in range(4):
    _, _, peaks = vc.calibrate_time_map(traces, u_atk, j, anchors)
    anchor_peaks[j] = peaks
centers, quality = vc.calibrate_time_map_dense(traces, u_atk, anchors, anchor_peaks)
vc.save_time_map(centers, quality, "project v3/time_map.json")
```

**통과 조건**: `|rho|` 중앙값 0.5 이상, 0.3 미만인 계수가 없을 것.
기준 실측값은 중앙값 0.6686 / 최소 0.5198 / 0.3 미만 0개다.
이보다 크게 나쁘면 매핑이 틀렸을 가능성이 높다.

**주의**: 선형 적합을 쓰지 말 것. poly1은 계수 간격이 앞부분 35에서 뒷부분 30으로
바뀌어 직선으로는 145샘플까지 어긋난다. 반드시 dense 방식을 쓴다.

---

## Step 3. 전처리 (정규화 memmap 생성)

원본 int64 12.2 GB를 정규화된 float16 3.2 GB로 한 번만 변환한다.
**시간이 오래 걸리므로 이미 있으면 건너뛴다.**

```python
import v3_normalize as nz, v3_stage2_data as s2

cfg = nz.NormConfig()   # 기본: MA=7, per_trace_gain=True, vertical_source="own"
prof_path = f"{WORK}/prof_norm.npy"
if not os.path.exists(prof_path):
    factory = lambda: vc.iter_profiling_traces(DATASET)
    s2.build_normalized_memmap(factory, prof_path, cfg, vc.N_PROFILING)

atk_path = f"{WORK}/atk_norm.npy"
if not os.path.exists(atk_path):
    def atk_factory():
        arr = vc.load_array(DATASET, "attack_10000traces.npy")
        def it():
            for i in range(0, vc.N_ATTACK, 1000):
                yield 1, i, np.asarray(arr[i:i+1000], dtype=np.float32)
        return it()
    s2.build_normalized_memmap(atk_factory, atk_path, cfg, vc.N_ATTACK)
```

**MA 폭을 바꾸고 싶다면** 실측 근거는 다음과 같다 (|rho| 평균).
```
MA=1  0.6653   MA=3  0.7312   MA=5  0.7434
MA=7  0.7589(최적)  MA=9  0.7442   MA=11 0.7192   MA=15 0.6650
```

**통과 조건**: 두 memmap 파일 생성, shape이 각각 (40000, 40000) / (10000, 40000).

---

## Step 4. 캠페인 간 이식성 측정

프로파일링과 공격 데이터가 다른 측정 세션이면 도메인 격차가 생긴다.
**v2는 이것을 재지 않고 프로파일링 통계를 공격셋에 이식했다.**

```python
Xp = np.load(prof_path, mmap_mode="r")
Xa = np.load(atk_path,  mmap_mode="r")
poi = centers.reshape(-1)                     # 1024개 계수 중심
port = nz.portability_report(np.asarray(Xp[:2000][:, poi]),
                             np.asarray(Xa[:2000][:, poi]))
```

**해석**
- `mean_shift_p95 > 0.1` 또는 `scale_ratio_median`이 0.9~1.1을 벗어나면
  캠페인 격차가 유의미하다 -> `vertical_source="own"` 유지가 필수다.
- 격차가 미미하면 `"external"`(v2 방식)도 안전하다는 뜻이며,
  `per_trace_gain`의 가치도 그만큼 낮다.

**이 결과를 리포트에 반드시 포함한다.** 미실행 시 "미측정"으로 남긴다.

---

## Step 5. 학습

```python
import v3_model as vm

u_prof = vc.load_array(DATASET, "profiling_40000_u=cs.npy")
hw     = s2.precompute_hw_labels(u_prof)
cm     = s2.build_time_map(centers)
train_idx, val_idx = s2.set_split_indices(val_set=4)   # leave-one-set-out

train = s2.CoefficientWindowSampler(Xp, hw, cm, train_idx, window=32,
                                    batch_size=512, traces_per_batch=32,
                                    shift_aug=1, seed=0)
val   = s2.CoefficientWindowSampler(Xp, hw, cm, val_idx, window=32,
                                    batch_size=512, traces_per_batch=32,
                                    shift_aug=0, seed=1)

model = vm.compile_model(vm.build_model(window=32, metal_safe=True))
model.summary()

history = model.fit(
    s2.to_tf_dataset(train),
    validation_data=s2.to_tf_dataset(val),
    steps_per_epoch=min(train.steps_per_epoch(), 2000),   # 1에폭이 3천만 샘플이라 제한
    validation_steps=200,
    epochs=50,
    callbacks=vm.make_callbacks(f"{WORK}/v3_model.keras"),
)
```

**`shift_aug` 선택 근거** (실측 |rho|): 0칸 0.7399 / 1칸 0.7236 / 3칸 0.6647 /
5칸 0.5656. 시간 매핑이 정확하므로 **1~2칸을 권장**한다.

**주의사항**
- `class_weight`를 쓰지 말 것. focal loss와 이중 적용되면 확률이 왜곡되고
  Step 6-2가 무너진다.
- 검증 정확도가 학습 정확도보다 높아도 놀라지 말 것. 드롭아웃 때문에 생기는
  정상적인 현상이며 "일반화의 증거"가 아니다.
- 손실이 NaN이 되면 `mixed_float16`을 껐는지 확인한다.

**통과 조건**: 검증 Top-1이 **8.68%를 크게 상회**할 것. 8.68% 근처면 학습이
전혀 안 된 것이다(모델이 최빈 클래스만 답하는 상태).

---

## Step 6. 평가 (3단계)

```python
import v3_report as rp, v3_evaluate as ev
```

### 6-1. 분류 성능

```python
POLY, COEFF = 0, 50
u_atk = vc.load_array(DATASET, "attack_10000_u=cs.npy")
atk_sampler = s2.CoefficientWindowSampler(Xa, s2.precompute_hw_labels(u_atk),
                                          cm, np.arange(vc.N_ATTACK), window=32,
                                          batch_size=512, traces_per_batch=32,
                                          shift_aug=0, seed=2)
probs = vm.predict_coefficient_probs(model, atk_sampler,
                                     np.arange(vc.N_ATTACK), POLY, COEFF)
y_atk = vc.hw32(u_atk[:, POLY, COEFF]).astype(np.int64)
cls_atk = rp.classification_report(probs, y_atk, label="공격셋")
```

검증셋에 대해서도 같은 방식으로 `cls_val`을 만든다.
**두 값의 차이가 이식성 격차다** (v2는 4.5%p였다).

### 6-1b. 계수별 편차 (계수 불변 모델의 핵심 검증)

```python
pairs = [(j, k) for j in range(4) for k in range(0, 256, 32)]
def predict_fn(j, k):
    p = vm.predict_coefficient_probs(model, atk_sampler, np.arange(2000), j, k)
    return p, vc.hw32(u_atk[:2000, j, k]).astype(np.int64)
per_coeff, coeff_rows = rp.per_coefficient_report(predict_fn, pairs)
```

**통과 조건**: 표준편차 5% 미만. 크면 시간 매핑이 일부 계수에서 어긋난 것이다.

### 6-2. 캘리브레이션

```python
rng  = np.random.default_rng(7)
perm = rng.permutation(len(y_atk))
cal, atk = perm[:3000], perm[3000:]          # 캘리브레이션 / 평가 분리
calib = rp.calibration_report(probs[atk], y_atk[atk])
```

**통과 조건**: ECE 0.10 미만. 크면 확률이 부정확해 Step 6-3이 신뢰할 수 없다.

### 6-3. 키 복구

```python
c_atk  = np.asarray(vc.load_array(DATASET, "attack_10000_c.npy"), dtype=np.int64)
s_true = vc.load_array(DATASET, "attack_s.npy")
attack = rp.attack_report(probs[cal], y_atk[cal], probs[atk], c_atk[atk],
                          COEFF, POLY, int(s_true[POLY, COEFF]),
                          u_true=np.asarray(u_atk)[atk],
                          n_candidates=65536, n_experiments=100, max_traces=40)
```

**해석 기준**: 완벽 오라클은 6개 트레이스에서 SR 100%에 도달한다.
어떤 모델도 이보다 잘할 수 없다. `efficiency`가 그 비율이다.

**절대 하지 말 것**
- 후보군을 `attack_s.npy`에서 만들지 말 것 (v2의 치명적 오류, 32,736배 축소)
- 온도를 공격셋에서 고르지 말 것
- 동점 처리를 빼지 말 것 (GE가 0으로 붕괴해 성공한 것처럼 보인다)

---

## Step 7. 리포트 생성

```python
meta = {
    "모델": "coefficient-invariant CNN (window=32)",
    "학습 샘플": f"{train.samples_per_epoch():,}",
    "정규화": cfg.describe(),
    "검증 분할": "leave-one-set-out (set4)",
    "shift_aug": 1,
    "타깃 계수": f"poly{POLY} coeff{COEFF}",
    "실행 환경": f"{info['tf_version']}, GPU={info['gpu_devices']}",
}
path = rp.render_markdown(f"{WORK}/v3_report.md", meta, cls_val, cls_atk,
                          calib, attack, per_coeff=per_coeff, portability=port)
rp.save_json(f"{WORK}/v3_report.json", {"meta": meta, "cls_atk": cls_atk,
                                        "calib": calib, "attack": attack,
                                        "per_coeff": per_coeff, "portability": port})
```

리포트를 사용자에게 보여주고, **미실행/미달 항목을 명확히 구분해서** 요약한다.

---

## 실패 시 진단 순서

| 증상 | 먼저 확인할 것 |
|---|---|
| 정확도가 8.68% 근처 | 시간 매핑(Step 2). 윈도우가 엉뚱한 곳을 보고 있다 |
| 계수별 편차가 큼 | 특정 poly의 매핑 오류. poly1을 의심한다 |
| ECE가 큼 | `class_weight`를 썼는지 확인. 온도 보정 필요 |
| GE가 안 떨어짐 | 후보군 생성 방식, 동점 처리, `c` 인덱싱(`c[trace, k]`) |
| GE가 너무 빨리 0 | 동점 처리 누락 또는 후보군이 정답에서 나왔는지 확인 |
| 검증-공격 격차 큼 | `vertical_source="own"` 확인, Step 4 이식성 재측정 |
| 메모리 부족 | `traces_per_batch` 축소, memmap이 float16인지 확인 |
| 학습이 느림 | Metal 인식 여부, `traces_per_batch`(지역성), XLA off 확인 |
