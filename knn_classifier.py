import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing import evaluate_model
from tqdm import tqdm

def train_knn(X_train, X_test, y_train, y_test):
    """
    Training và đánh giá model KNN với thanh progress
    """
    print("\nTraining KNN model...")
    # Khởi tạo model KNN
    knn = KNeighborsClassifier(n_neighbors=5)

    # Đo thời gian training
    start_time = time.time()

    # Training model với thanh progress
    with tqdm(total=100, desc="Training KNN") as pbar:
        knn.fit(X_train, y_train)
        pbar.update(100)

    # Tính thời gian training
    training_time = time.time() - start_time

    print("\nEvaluating KNN model...")
    # Dự đoán trên tập test
    y_pred = knn.predict(X_test)

    # Đánh giá model
    accuracy, _ = evaluate_model("KNN", y_test, y_pred, training_time)

    return knn, accuracy, training_time

def predict_single_image_knn(model, image, scaler):
    """
    Dự đoán nhãn cho một ảnh đơn lẻ
    """
    # Chuẩn hóa ảnh
    image_scaled = scaler.transform(image.reshape(1, -1))

    # Dự đoán
    prediction = model.predict(image_scaled)

    return prediction[0]