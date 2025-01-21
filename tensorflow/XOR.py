import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. XOR 데이터 준비
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. 모델 정의
model = Sequential([
    Dense(4, input_dim=2, activation='relu'),  # 은닉층 (노드 4개)
    Dense(1, activation='sigmoid')            # 출력층 (노드 1개)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. 모델 훈련
model.fit(x_data, y_data, epochs=1000, verbose=0)

# 4. 결과 확인
predictions = model.predict(x_data)
print("예측값:")
print(np.round(predictions))  # 0.5 기준으로 반올림
