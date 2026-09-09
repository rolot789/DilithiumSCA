---
name: dilithium-v3-run
description: Dilithium v3 부채널 공격 모델을 처음부터 끝까지 실행하고 리포트를 생성한다. 환경 점검, 데이터 검증, 전처리, 시간 매핑 보정, 학습(단일/멀티태스크/앙상블), 3단계 평가, PI 기반 학습 방식 비교, 강한 CNN 기준 헤드 재선택, 마크다운 리포트까지 단계별로 진행하며 각 단계마다 통과 조건을 확인한다. 사용자가 "v3 학습 돌려줘", "모델 학습하고 리포트 뽑아줘", "이식성 측정해줘", "GE 재줘", "헤드 선택 다시 확인해줘", "앙상블 벤치마크 돌려줘" 같은 요청을 할 때 사용한다. Apple Silicon(M시리즈) 환경을 기본 가정한다.
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

WINDOW, OFFSET = 96, 24    # 실측 근거는 아래 참조
train = s2.CoefficientWindowSampler(Xp, hw, cm, train_idx, window=WINDOW,
                                    window_offset=OFFSET, batch_size=512,
                                    traces_per_batch=32, shift_aug=1, seed=0)
val   = s2.CoefficientWindowSampler(Xp, hw, cm, val_idx, window=WINDOW,
                                    window_offset=OFFSET, batch_size=512,
                                    traces_per_batch=32, shift_aug=0, seed=1)

model = vm.compile_model(vm.build_model(window=WINDOW, metal_safe=True))
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

**`window` / `window_offset` 선택 근거**: 계수 하나가 **6~7개 지점에서 누설**한다
(poly0/coeff50 기준 상대위치 -1, +15, +24, +36, +55, +64). `window=32`는 앞의 두
개만 담는다. 넓혀서 실측한 |rho|는 다음과 같다.

```
window= 32 offset= 0  0.7974 (기준)     window= 96 offset=24  0.8748 (+9.7%)
window= 64 offset=16  0.8587 (+7.7%)    window=128 offset=32  0.8966 (+12.4%)
                                        window=160 offset=40  0.9067 (+13.7%)
```

96~128이 권장 구간이다. 그 이상은 수확체감이고 이웃 계수와의 중첩이 커진다.

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

### 5b. 멀티태스크 변형 (정보량 상한 2.0배)

HW 하나만 예측하면 정보량 상한이 4.2147비트다. sign/byte0/byte1/byte2로 나누면
**결합 엔트로피 8.4157비트**로 1.997배가 되고, 완벽 모델 기준 필요 트레이스가
5.46개에서 2.73개로 준다.

주변 엔트로피의 단순 합(8.7853)을 쓰면 안 된다. 헤드가 완전히 독립일 때만
성립하는 상한이고 실제로는 0.3696비트(4.2%)가 중복이다.

```python
extra = s2.precompute_multitask_labels(u_prof)     # sign, byte0, byte1, byte2
train_mt = s2.CoefficientWindowSampler(Xp, hw, cm, train_idx, window=WINDOW,
                                       window_offset=OFFSET, batch_size=512,
                                       traces_per_batch=32, shift_aug=1, seed=0,
                                       extra_labels=extra)
mt = vm.compile_multitask(vm.build_multitask_model(window=WINDOW))
```

**중요: 헤드를 미리 쳐내지 말고 4개 전부로 학습한다.** 어떤 헤드를 쓸지는
학습이 끝난 뒤 Step 6-e에서 이 모델의 실측치로 결정한다. 선형 프로브 기준으로는
`sign` 단독이 가장 좋았지만, 그것은 프로브가 약한 바이트를 학습하지 못했기
때문일 수 있다. **강한 CNN에서는 결론이 뒤집힐 수 있다.**

byte3은 sign과 완전 중복이라(결합 엔트로피 기여 +0.0000비트) 기본 헤드에서 제외되어 있다.

---

### 5c. 앙상블 (선택)

```python
import v3_ensemble as en

specs = en.recommended_specs()      # 넓은 창 1개 + disjoint 보조 4개
members = []
for i, sp in enumerate(specs):
    tr = s2.CoefficientWindowSampler(Xp, hw, cm, train_idx, batch_size=512,
                                     traces_per_batch=32, shift_aug=1, seed=i, **sp)
    va = s2.CoefficientWindowSampler(Xp, hw, cm, val_idx, batch_size=512,
                                     traces_per_batch=32, shift_aug=0, seed=100+i, **sp)
    m = vm.compile_model(vm.build_model(window=sp["window"]))
    m.fit(s2.to_tf_dataset(tr), validation_data=s2.to_tf_dataset(va),
          steps_per_epoch=1000, validation_steps=100, epochs=30,
          callbacks=vm.make_callbacks(f"{WORK}/ens{i}.keras"))
    members.append((m, sp))

ens = en.Ensemble(members)
factory = lambda sp: s2.CoefficientWindowSampler(Xa, hw_atk, cm, np.arange(vc.N_ATTACK),
                                                 batch_size=512, traces_per_batch=32,
                                                 shift_aug=0, seed=2, **sp)
_, member_probs = ens.predict(factory, np.arange(vc.N_ATTACK), POLY, COEFF)
ens.calibrate([p[cal] for p in member_probs], y_atk[cal])    # 공격셋 미사용
probs_ens = en.combine_probs(member_probs, ens.mode, ens.weights)
print(en.ensemble_gain([p[atk] for p in member_probs], y_atk[atk], ens.mode, ens.weights))
en.diversity_report([p[atk] for p in member_probs], y_atk[atk])
```

**다양성 구성 선택 근거** (선형 프로브 실측):

| 구성 | 오차 상관 | 앙상블 이득 |
|---|---|---|
| 배깅 (같은 창) | 0.946 | 1.003배 |
| 중첩 창 (폭만 확대) | 0.432 | 1.018배 |
| disjoint 창 | 0.258 | 1.195배 |
| 넓은 창 + disjoint | - | 1.108배 (절대 PI 최고) |

**중첩 창(폭만 다른 창)을 멤버로 쓰지 말 것.** 넓은 창이 좁은 창을 포함해
우열 관계가 되고 가중치가 한쪽에 몰린다. 폭 확대는 단일 모델 설정으로 쓴다.

`diversity_report`의 오차 상관이 0.8을 넘으면 멤버가 너무 비슷한 것이므로
구성을 바꾼다. 다만 **선형 프로브 수치는 신경망을 과소평가**하므로
배깅 1.003배를 "시드 앙상블 무용"으로 읽지 말고 PI로 직접 확인한다.

모델을 여러 개 학습하기 부담스러우면 `en.make_snapshot_callback()`으로
한 번의 학습에서 스냅샷 앙상블을 얻는다(cyclic LR, 비용 1/N).

### 5d. Ranking Loss (실험용, 기본은 쓰지 않는다)

`v3_ranking.py`에 구현되어 있고 기울기까지 검증했지만(유한차분 오차 2.68e-08),
**실측에서 CE보다 나빴다.**

| 손실 | Top-1 | PI(보정 전) | PI(보정 후) |
|---|---|---|---|
| CE 단독 | 20.39% | 0.691 | **0.710** |
| CE + RkL | 14.89% | -10.143 | 0.215 |
| RkL 위주(ce=0.2) | 15.10% | -20.119 | -0.219 |

RkL은 누적 점수의 상대 순서만 제약해 로그확률의 절대 크기를 붙잡지 않는다.
그래서 확신을 갖고 틀리는 상태가 되고 PI가 음수로 붕괴한다. 우리 공격은
로그확률을 누적하므로 캘리브레이션이 곧 성능이다.

굳이 시도한다면:

```python
import v3_ranking as rk
s_star, c_vals = rk.synthesize_key_assignment(u_prof, POLY, COEFF, seed=0)
print(rk.verify_assignment(c_vals, s_star, u_prof, POLY, COEFF))  # 잔여류 1.0 확인
sam = rk.RankingBatchSampler(Xp, hw, u_prof, cm, train_idx, window=WINDOW,
                             window_offset=OFFSET, n_groups=8, group_size=16,
                             n_candidates=64, shift_aug=1, seed=0)
m = rk.make_ranking_model(vm.build_model(window=WINDOW), alpha=1.0, ce_weight=1.0)
m.compile(optimizer=bk.make_optimizer(1e-3), **bk.compile_kwargs())
m.fit(rk.to_tf_dataset(sam, WINDOW), steps_per_epoch=1000, epochs=20)
```

**반드시** `ce_weight >= 1.0`, 학습 후 온도 보정, 그리고 PI를 CE 단독과 비교한다.
PI가 낮으면 채택하지 않는다.

프로파일링에 c/s가 없어 `synthesize_key_assignment()`로 (가짜키, challenge)를
역산해 쓴다. 학습용 후보에 정답을 포함하는 것은 지도학습 라벨에 해당하므로
정당하다 — **평가용 후보는 반드시 `v3_evaluate.build_candidate_residues()`** 를 쓴다.

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

### 6-d. 학습 방식 간 효율 비교 (PI)

여러 학습 방식을 시도했다면 **PI(Perceived Information)** 로 비교한다.
정확도는 라벨 종류가 다르면 비교 자체가 불가능하지만 PI는 같은 비트 단위다.

```python
import v3_benchmark as bm

# 단일 모델
v = bm.Variant("멀티태스크 w96", "sign+byte0~2, window=96", scheme="multitask", window=96)
m = bm.evaluate_variant(v, probs, y_atk, n_params=model.count_params(),
                        train_seconds=elapsed, train_samples=train.samples_per_epoch(),
                        sr100=attack["traces_to_sr100"], oracle_sr100=6)
all_metrics.append(m)

# 앙상블은 멤버 확률 리스트를 넘긴다. 다양성과 이득이 자동으로 채워진다.
ve = bm.Variant("앙상블 넓은창+disjoint", "", window=128,
                ensemble={"source": "mixed"})     # bagging|nested|disjoint|mixed|snapshot
me = bm.evaluate_ensemble_variant(ve, [p[atk] for p in member_probs], y_atk[atk],
                                  ens.mode, ens.weights,
                                  n_params=sum(m.count_params() for m, _ in members),
                                  train_seconds=elapsed_all)
all_metrics.append(me)

bm.render_benchmark(f"{WORK}/benchmark.md", all_metrics)
```

멀티태스크는 헤드가 여러 개라 전용 함수를 쓴다. Top-1은 **모든 헤드가 동시에
맞은 비율**로 재어 단일 헤드 모델과 비교가 성립하게 한다.

```python
vm_ = bm.Variant("멀티태스크 w128", "sign+byte0~2", window=128)
mm = bm.evaluate_multitask_variant(vm_, head_probs, head_labels,
                                   n_params=model.count_params(),
                                   train_seconds=elapsed)
all_metrics.append(mm)
```

멀티태스크 리포트에는 **헤드 분해 표**(단독 PI_h, 한계 기여, "빼면?" 판정)가
들어간다. **어느 헤드를 쓸지는 여기서 정하지 말고 Step 6-e에서 결정한다.**

**앙상블을 멀티태스크로 구성할 때**는 헤드별로 결합한다. 멤버마다 잘하는 헤드가
다르기 때문이다(실측: sign은 넓은 창 멤버에 가중치 0.966이 몰렸지만,
byte0은 [0.348, 0.085, 0.229, 0.274, 0.064]로 disjoint 멤버들에 퍼졌다).

```python
sel = en.select_combine_mode_multitask(member_head_probs_cal, head_labels_cal)
mm = bm.evaluate_multitask_ensemble_variant(v, member_head_probs_atk, head_labels_atk,
                                            sel["mode"], sel["per_head_weights"])
```

리포트에는 **앙상블 다양성 표**도 별도로 들어간다(다양성 원천, 멤버 수, 결합 방식,
오차 상관, 불일치율, 최고 멤버 PI 대비 이득). 오차 상관이 0.8을 넘으면 `!`가 붙는데,
멤버들이 사실상 같은 모델이라는 뜻이므로 구성을 바꾼다.

`ensemble={"source": ...}`에 원천을 적어두면 `DIVERSITY_REFERENCE`의 실측 기준값
(배깅 1.003x / 중첩창 1.018x / disjoint 1.195x / 넓은창+disjoint 1.108x)과
나란히 비교된다.

**해석**
- `pi_bits`: 트레이스 하나에서 실제로 뽑아낸 정보량. 클수록 좋다.
- `predicted_traces` = 23.0 / PI. 실측 SR100과 대조하면 지표 신뢰도를 알 수 있다
  (검증 시 예측/실측 비 1.02였다).
- **`pi_bits`가 음수면 즉시 중단한다.** 모델이 확신을 갖고 틀리는 상태이며,
  정보를 주는 게 아니라 뺏고 있다. 정확도로는 이 상태가 안 보인다.
- `pi_ratio` = PI / H(Y). 그 라벨의 이론적 상한에 얼마나 근접했는지다.

기준값 (라벨 자체의 정보량 상한, 공격셋 u 전량 10,240,000 샘플 실측):

| 라벨 | 엔트로피 | 완벽 모델의 필요 트레이스 |
|---|---|---|
| HW 단독 (33클래스) | 4.2147 | 5.46 |
| 멀티태스크 결합 (4헤드) | 8.4157 | 2.73 |

주변 엔트로피의 합(8.7853)은 헤드가 완전히 독립일 때만 성립하는 상한이므로
쓰지 않는다. 실제 중복이 0.3696비트(4.2%)다.

---

### 6-e. 강한 CNN 기준 헤드 재선택 (멀티태스크를 쓴 경우 필수)

**기존 헤드 결론을 그대로 물려받지 말 것.** 지금까지 기록된
"`sign` 단독이 최선"은 **선형 프로브로 잰 값**이며, 그 프로브가 byte0/byte1을
전혀 학습하지 못한 결과일 가능성이 크다. 결합 엔트로피 8.4157비트 중 **5.09비트가
byte0/byte1에 있으므로**, CNN이 이 둘을 학습해내면 결론이 뒤집힌다.
이 단계의 목적은 그것을 실제 모델로 판정하는 것이다.

#### 절차

```python
import v3_benchmark as bm

# 1) 헤드 4개 전부로 학습된 모델에서 헤드별 확률을 뽑는다 (미리 쳐내지 않았어야 한다)
head_probs = {h: mt.predict(X_windows, verbose=0)[i]
              for i, h in enumerate(mt.output_names)}
head_labels = {h: bm.SCHEMES[h](u_atk_sel).astype("int64") for h in head_probs}

# 2) 헤드별 온도 보정 (반드시 캘리브레이션셋에서)
import v3_evaluate as ev
for h in head_probs:
    T, _ = ev.fit_temperature(head_probs[h][cal], head_labels[h][cal])
    head_probs[h] = ev.apply_temperature(head_probs[h], T)
    print(f"  {h}: T={T:.2f}")     # T가 0.05나 30에 닿으면 그리드 경계 문제다

# 3) 한계 기여로 판정 (단독 PI가 아니다)
abl = bm.head_ablation({h: p[cal] for h, p in head_probs.items()},
                       {h: y[cal] for h, y in head_labels.items()})
for h, d in abl["heads"].items():
    print(f"  {h}: 한계 기여 {d['marginal']:+.3f}  "
          f"{'빼는 게 이득' if d['drop_is_better'] else '유지'}")

# 4) 탐욕적 선택 — 반드시 캘리브레이션셋에서
chosen = bm.select_heads({h: p[cal] for h, p in head_probs.items()},
                         {h: y[cal] for h, y in head_labels.items()})
print("선택된 헤드:", chosen["heads"])
```

#### 판정 기준

선형 프로브가 남긴 기준선이다. **CNN이 이 값을 넘는지가 핵심 질문이다.**

| 항목 | 선형 프로브 | CNN이 이러면 결론이 바뀐다 |
|---|---|---|
| byte0 Top-1 | 28.03% | 유의미하게 상회 |
| byte1 Top-1 | 27.59% | 유의미하게 상회 |
| byte0 한계 기여 | -0.058 | **양수** |
| byte1 한계 기여 | -0.053 | **양수** |
| byte2 한계 기여 | -0.140 | **양수** |
| 최선 조합 결합 PI | 0.660 (`sign` 단독) | 4헤드가 이보다 높으면 4헤드 채택 |

- **한계 기여가 양수인 헤드는 모두 유지한다.** 단독 PI_h가 양수라도 한계 기여가
  음수면 뺀다(byte2가 실제로 그 사례였다: 단독 +0.241, 한계 -0.140).
- `select_heads`가 4헤드를 그대로 유지하면, 멀티태스크가 제 값을 하는 것이다.
  이때 예측 트레이스가 HW 단독(5.46) 쪽으로 크게 줄어드는지 확인한다.
- `sign` 단독으로 다시 수렴하면, **CNN도 약한 바이트를 학습하지 못한 것**이다.
  이 경우 멀티태스크를 포기하기 전에 다음을 먼저 시도한다.
  - `compile_multitask(loss_weights=...)`로 약한 헤드의 가중치를 **올려** 본다
  - 윈도우를 넓혀 본다(byte0/byte1의 누설 지점이 다른 곳일 수 있다)
  - 앙상블과 결합해 본다. 실측에서 앙상블이 약한 헤드를 특히 보강했다
    (byte0 가중치가 disjoint 멤버들로 퍼짐)

#### 학습용 헤드와 공격용 헤드는 다를 수 있다

이 구분을 놓치기 쉽다.

- **학습용**: 손실에 들어가는 헤드. 약한 헤드도 공유 트렁크에 유용한 보조 신호를
  줄 수 있어 남겨두는 편이 나을 수 있다.
- **공격용**: 로그우도를 합산할 헤드. 여기서는 한계 기여가 음수인 헤드를 뺀다.

즉 **4헤드로 학습하고 선택된 부분집합으로 공격**하는 구성이 가능하며,
대개 이쪽이 낫다. 헤드를 아예 빼고 재학습하는 것은 `select_heads` 결과가
안정적으로 같은 답을 줄 때만 한다. 두 방식을 모두 벤치마크에 올려 비교한다.

```python
# 학습은 4헤드, 공격은 선택된 부분집합
hs = chosen["heads"]
v = bm.Variant(f"멀티태스크 4헤드 학습 / {'+'.join(hs)} 공격", "", window=WINDOW)
bm.evaluate_multitask_variant(v, {h: head_probs[h][atk] for h in hs},
                              {h: head_labels[h][atk] for h in hs}, head_names=hs)
```

#### 통과 조건

- 헤드별 온도가 그리드 경계(0.05 또는 30)에 닿지 않을 것
- 선택된 조합의 결합 PI가 **HW 단독 라벨보다 높을 것**. 낮으면 멀티태스크를
  쓸 이유가 없으므로 HW 단독으로 돌아간다.
- 결과를 `evaluate_multitask_variant`로 벤치마크에 올려 다른 방식과 함께 기록

#### 결과 기록 (이 단계의 마지막, 건너뛰지 말 것)

```python
bm.baseline_from_ablation(
    "cnn",
    {h: p[cal] for h, p in head_probs.items()},     # 캘리브레이션셋 확률
    {h: y[cal] for h, y in head_labels.items()},
    model_info={"window": WINDOW, "window_offset": OFFSET,
                "params": mt.count_params(), "epochs": len(history.history["loss"])},
    notes="여기에 관찰한 것을 적는다 (예: byte0 학습 성공 여부)")
```

**이 한 번의 호출로 다음이 전부 끝난다.**

1. 한계 기여 계산 (`head_ablation`)
2. 헤드 선택 (`select_heads`)
3. `project v3/head_baseline.json` 기록 — 이전 값은 `history`에 보존되어
   선형 프로브 -> CNN 변화를 추적할 수 있다
4. **`instruction_v3.md` 10.3절의 실측값 블록 자동 갱신**
   (`AUTO:head-baseline` 마커 사이만 교체되므로 손으로 쓴 설명은 보존된다)
5. 이후 모든 리포트의 "멀티태스크 헤드 분해" 표가 CNN 기준값을 사용

문서만 다시 맞추고 싶으면 `bm.sync_docs()`를 호출한다.

**갱신 대상을 혼동하지 말 것.**

| 대상 | 성격 | CNN 결과로 갱신? |
|---|---|---|
| `head_baseline.json` | 헤드별 PI / 한계 기여 / 선택된 헤드 | **자동 갱신** |
| `instruction_v3.md` 10.3절 AUTO 블록 | 위 값의 문서 표현 | **자동 갱신** |
| `v3_benchmark.MULTITASK_REFERENCE` | 라벨 자체의 엔트로피 | **갱신하지 않는다** |

`MULTITASK_REFERENCE`의 값(결합 엔트로피 8.4157, HW 4.2147 등)은 `u` 분포의
성질을 1,024만 샘플로 잰 것이라 **모델과 무관하다.** 어떤 CNN을 학습해도 바뀌지
않으므로 건드리지 않는다. 여기를 고치려 든다면 무언가 잘못 이해한 것이다.

AUTO 블록 **안의 내용을 손으로 고치지 말 것.** 다음 호출에서 덮어쓰인다.
설명을 덧붙이려면 마커 바깥에 쓰거나 `notes=` 인자로 넘긴다.

---

## Step 7. 리포트 생성

```python
meta = {
    "모델": f"coefficient-invariant CNN (window={WINDOW}, offset={OFFSET})",
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

# 학습 방식을 여러 개 비교했다면 벤치마크도 함께
bm.render_benchmark(f"{WORK}/benchmark.md", all_metrics)
```

### 7b. 문서 동기화 (마지막, 항상 실행)

```python
bm.sync_docs()      # instruction_v3.md 10.3절 + README.md 5.4절
```

레포의 사양서와 README에 있는 **헤드 실측값 AUTO 블록**을 현재
`head_baseline.json`과 맞춘다. Step 6-e에서 `baseline_from_ablation()`을
호출했다면 이미 갱신되어 있지만, **항상 한 번 더 실행한다.** 이유는 두 가지다.

- 멱등이라 부작용이 없다. 이미 맞으면 "변경 없음"만 출력한다.
- 누군가 AUTO 블록 안을 손으로 고쳤거나, 다른 경로로 기준값이 바뀐 경우를 잡는다.

출력이 "갱신"으로 나오면 **문서가 실제로 바뀐 것이므로 커밋 대상**이다.
사용자에게 어느 파일이 바뀌었는지 알린다.

```
갱신: .../instruction_v3.md      <- 커밋 필요
변경 없음: .../README.md
```

`AUTO:head-baseline` 마커가 없는 문서는 조용히 건너뛴다. 마커 바깥의 손으로 쓴
설명은 절대 건드리지 않는다.

### 7c. 마무리 점검

리포트를 사용자에게 보여주고 다음을 명확히 구분해 요약한다.

- **미실행 항목** — 수치를 지어내지 말고 "미실행"으로 남긴다
- **미달 항목** — 통과 조건을 못 넘은 단계
- **문서 갱신 여부** — `sync_docs()`가 바꾼 파일 목록
- **기준값 출처** — `head_baseline.json`의 `source`가 아직 `linear_probe`면,
  CNN 실측치로 갱신되지 않았다는 뜻이므로 그 사실을 반드시 알린다

```python
b = bm.load_head_baseline()
print("헤드 기준값 출처:", b["source"], "| 선택된 헤드:", b["chosen_heads"])
```

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
| 멀티태스크가 HW 단독보다 나쁨 | Step 6-e로 헤드 재선택. 한계 기여가 음수인 헤드를 공격에서 뺀다 |
| `select_heads`가 `sign` 단독으로 수렴 | CNN도 약한 바이트 학습 실패. 약한 헤드 `loss_weights` 상향, 윈도우 확대, 앙상블 결합을 먼저 시도 |
| 헤드 온도가 0.05나 30에 닿음 | 그리드 경계 문제. 보정이 덜 된 상태라 PI가 왜곡된다 |
