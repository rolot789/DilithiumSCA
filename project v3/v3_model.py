"""계수 불변(coefficient-invariant) 프로파일링 모델.

v2 대비 달라진 점
  * 입력이 "떨어진 윈도우 4개를 이어붙인 60샘플"이 아니라 **연속된 단일 윈도우**다.
    v2는 15샘플마다 인위적 경계가 있는 배열에 kernel=11 컨볼루션을 걸어 무관한
    시간 구간을 섞었다. 여기서는 커널이 물리적으로 연속인 구간만 본다.
  * class_weight를 쓰지 않는다. focal loss와 이중 적용되면 사후확률이 왜곡되어
    로그우도 누적 공격이 망가진다. 캘리브레이션은 학습 후 온도로 처리한다.
  * 증강은 GaussianNoise가 아니라 **랜덤 시간 이동**이다(샘플러가 담당).
    이미 단위분산인 데이터에 sigma=0.05 노이즈는 사실상 아무 효과가 없다.
  * 하나의 가중치가 1024개 계수 전부를 담당하므로 Stage 3(전체 키 복구)이
    별도 작업이 아니라 이 모델의 부산물이 된다.
"""

import numpy as np

from v3_common import N_HW_CLASSES


def _conv(x, filters, kernel, metal_safe, name=None):
    """Metal 백엔드에서 Conv1D 대신 (k,1) Conv2D 경로를 쓸 수 있게 한다."""
    from tensorflow.keras import layers
    if metal_safe:
        return layers.Conv2D(filters, (kernel, 1), padding="same",
                             use_bias=False, name=name)(x)
    return layers.Conv1D(filters, kernel, padding="same", use_bias=False, name=name)(x)


def _pool(x, metal_safe):
    from tensorflow.keras import layers
    return layers.MaxPooling2D((2, 1))(x) if metal_safe else layers.MaxPooling1D(2)(x)


def _residual_block(x, filters, kernel, metal_safe, dropout=0.1):
    from tensorflow.keras import layers
    shortcut = x
    h = _conv(x, filters, kernel, metal_safe)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.SpatialDropout2D(dropout)(h) if metal_safe else layers.SpatialDropout1D(dropout)(h)
    h = _conv(h, filters, kernel, metal_safe)
    h = layers.BatchNormalization()(h)
    if shortcut.shape[-1] != filters:
        shortcut = _conv(shortcut, filters, 1, metal_safe)
        shortcut = layers.BatchNormalization()(shortcut)
    h = layers.Add()([h, shortcut])
    h = layers.Activation("relu")(h)
    return _pool(h, metal_safe)


def build_model(window, n_classes=N_HW_CLASSES, width=32, n_blocks=3,
                kernel=7, dropout=0.1, head_dropout=0.3, metal_safe=True):
    """윈도우 하나 -> HW 분포. 모든 계수가 같은 가중치를 공유한다."""
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(window, 1), name="window")
    x = layers.Reshape((window, 1, 1))(inp) if metal_safe else inp

    x = _conv(x, width, kernel, metal_safe, name="stem")
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for b in range(n_blocks):
        x = _residual_block(x, width * (2 ** b), kernel, metal_safe, dropout)

    x = layers.GlobalAveragePooling2D()(x) if metal_safe else layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(head_dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", dtype="float32", name="hw")(x)
    return Model(inp, out, name="coeff_invariant_cnn")


def build_multitask_model(window, heads=None, width=32, n_blocks=3,
                          kernel=7, dropout=0.1, head_dropout=0.3,
                          metal_safe=True, include_hw=True):
    """공유 트렁크 + 헤드 여러 개.

    HW 하나만 예측하면 트레이스당 4.215비트가 상한이다. 바이트별 HW를 나눠
    예측하면 헤드 합이 8.78비트로 약 2.1배가 되고, 필요한 트레이스 수가 절반이 된다.
    (v3_benchmark의 실측 근거 참조)

    헤드끼리 정보가 겹치므로 합이 그대로 실현되지는 않는다. 반드시 PI와 실측 GE로
    검증할 것.
    """
    from tensorflow.keras import layers, Model
    from v3_benchmark import MULTITASK_HEADS, SCHEMES

    heads = heads or MULTITASK_HEADS
    inp = layers.Input(shape=(window, 1), name="window")
    x = layers.Reshape((window, 1, 1))(inp) if metal_safe else inp
    x = _conv(x, width, kernel, metal_safe, name="stem")
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    for b in range(n_blocks):
        x = _residual_block(x, width * (2 ** b), kernel, metal_safe, dropout)
    x = layers.GlobalAveragePooling2D()(x) if metal_safe else layers.GlobalAveragePooling1D()(x)
    trunk = layers.Dense(128, activation="relu", name="trunk")(x)
    trunk = layers.Dropout(head_dropout)(trunk)

    outs = {}
    for h in heads:
        outs[h] = layers.Dense(SCHEMES[h].n_classes, activation="softmax",
                               dtype="float32", name=h)(trunk)
    if include_hw:
        outs["hw"] = layers.Dense(N_HW_CLASSES, activation="softmax",
                                  dtype="float32", name="hw")(trunk)
    return Model(inp, outs, name="coeff_invariant_multitask")


def compile_multitask(model, learning_rate=1e-3, label_smoothing=0.05,
                      loss_weights=None):
    """헤드별 sparse CE. class_weight는 쓰지 않는다(확률이 왜곡된다)."""
    import tensorflow as tf
    from v3_backend import compile_kwargs, make_optimizer

    losses = {n: tf.keras.losses.SparseCategoricalCrossentropy()
              for n in model.output_names}
    model.compile(optimizer=make_optimizer(learning_rate), loss=losses,
                  loss_weights=loss_weights,
                  metrics={n: "accuracy" for n in model.output_names},
                  **compile_kwargs())
    return model


def compile_model(model, learning_rate=1e-3, label_smoothing=0.05):
    """class_weight 없이 표준 CE + label smoothing. Metal에서는 XLA를 끈다."""
    import tensorflow as tf
    from v3_backend import compile_kwargs, make_optimizer

    model.compile(
        optimizer=make_optimizer(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")],
        **compile_kwargs(),
    )
    return model


def make_callbacks(checkpoint_path="v3_coeff_invariant.keras", patience=8):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss",
                                           save_best_only=True),
    ]


def predict_coefficient_probs(model, sampler, trace_indices, poly, coeff,
                              batch_size=256):
    """Stage 1 평가로 넘길 (n_traces, 33) 확률 행렬.

    계수 하나에 대해 지정된 공격 트레이스들의 예측을 모은다. 이 출력을
    v3_evaluate.fit_temperature / trace_log_likelihood / guessing_entropy에
    그대로 넣으면 정직한 GE/SR이 나온다.
    """
    windows = np.concatenate([
        sampler.full_coefficient_batch(ti, poly, [coeff]) for ti in trace_indices
    ])
    return model.predict(windows, batch_size=batch_size, verbose=0)


def evaluate_accuracy(model, sampler, n_batches=200):
    """다수 클래스 베이스라인(8.68%)과 함께 정확도를 보고한다."""
    gen = sampler.batches()
    correct = total = 0
    hist = np.zeros(N_HW_CLASSES, dtype=np.int64)
    for _ in range(n_batches):
        xb, yb = next(gen)
        pred = model.predict(xb, verbose=0).argmax(axis=1)
        correct += int((pred == yb).sum())
        total += len(yb)
        hist += np.bincount(yb, minlength=N_HW_CLASSES)
    acc = correct / total
    baseline = hist.max() / hist.sum()
    print(f"  Top-1 정확도 {acc*100:.2f}%  |  다수 클래스 베이스라인 {baseline*100:.2f}%"
          f"  |  배율 {acc/baseline:.2f}x")
    return {"accuracy": acc, "majority_baseline": baseline}
