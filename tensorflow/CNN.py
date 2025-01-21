import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 데이터 전처리
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 구성
model = models.Sequential([
    # 첫 번째 컨볼루션 레이어
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 두 번째 컨볼루션 레이어
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),

    # 세 번째 컨볼루션 레이어
    layers.Conv2D(64, (3, 3), activation='relu'),

    # 플래튼 레이어
    layers.Flatten(),

    # 완전 연결 레이어 (Fully Connected Layer)
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 출력 레이어
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 훈련
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 정확도: {test_acc}")
