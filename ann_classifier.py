import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from data_preprocessing import evaluate_model
from tqdm import tqdm

def train_ann(X_train, X_test, y_train, y_test):
    """
    Training và đánh giá model ANN với thanh progress
    """
    print("\nTraining ANN model...")
    # Khởi tạo model ANN
    ann = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 hidden layers với 100 và 50 neurons
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        verbose=False  # Tắt verbose mặc định của MLPClassifier
    )

    # Đo thời gian training
    start_time = time.time()

    # Training model với thanh progress
    with tqdm(total=100, desc="Training ANN") as pbar:
        ann.fit(X_train, y_train)
        pbar.update(100)

    # Tính thời gian training
    training_time = time.time() - start_time

    print("\nEvaluating ANN model...")
    # Dự đoán trên tập test
    y_pred = ann.predict(X_test)

    # Đánh giá model
    accuracy, _ = evaluate_model("ANN", y_test, y_pred, training_time)

    return ann, accuracy, training_time

def predict_single_image_ann(model, image, scaler):
    """
    Dự đoán nhãn cho một ảnh đơn lẻ
    """
    # Chuẩn hóa ảnh
    image_scaled = scaler.transform(image.reshape(1, -1))

    # Dự đoán
    prediction = model.predict(image_scaled)

    return prediction[0]