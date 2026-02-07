import numpy as np
import os
import tensorflow as tf

# 1. 경로 및 파일 확인
DATASET_DIR = './Dataset'
RHO_PATH = './Pearson correlation coefficient/rho.npy'
LABEL_PATH = os.path.join(DATASET_DIR, 'profiling_40000_u=cs.npy')

def get_hw(n):
    return bin(n & 0xffffffff).count('1')

def verify():
    print("--- 1. 데이터 로딩 테스트 ---")
    try:
        labels_u = np.load(LABEL_PATH)
        rho = np.load(RHO_PATH)
        print(f"Labels loaded: {labels_u.shape}")
        print(f"Rho loaded: {rho.shape}")
    except Exception as e:
        print(f"로드 실패: {e}")
        return

    print("\n--- 2. HW 및 POI 추출 테스트 ---")
    # rho: (20, 40000)
    avg_abs_rho = np.mean(np.abs(rho), axis=0)
    max_rho = np.max(avg_abs_rho)
    print(f"Max Average Absolute Rho: {max_rho}")
    
    threshold = 0.03
    pois = np.where(avg_abs_rho > threshold)[0]
    print(f"임계값 {threshold} 기준 POI 개수: {len(pois)}")
    
    if len(pois) == 0:
        print("POI가 없습니다. 상위 100개를 선택합니다.")
        pois = np.argsort(avg_abs_rho)[-100:]
        print(f"선택된 상위 100개 POI 인덱스: {pois[:10]}...")

    print("\n--- 3. 모델 생성 테스트 ---")
    input_dim = len(pois)
    num_classes = 33
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(f"입력 차원 {input_dim}으로 모델 컴파일 성공")

if __name__ == "__main__":
    verify()
