import time
from sklearn.svm import SVC
from data_preprocessing import evaluate_model
from tqdm import tqdm

def train_svm(X_train, X_test, y_train, y_test):
    """
    Training và đánh giá model SVM với thanh progress
    """
    print("\nTraining SVM model...")
    # Khởi tạo model SVM
    svm = SVC(kernel='rbf', random_state=42)

    # Đo thời gian training
    start_time = time.time()

    # Training model với thanh progress
    with tqdm(total=100, desc="Training SVM") as pbar:
        svm.fit(X_train, y_train)
        pbar.update(100)

    # Tính thời gian training
    training_time = time.time() - start_time

    print("\nEvaluating SVM model...")
    # Dự đoán trên tập test
    y_pred = svm.predict(X_test)

    # Đánh giá model
    accuracy, _ = evaluate_model("SVM", y_test, y_pred, training_time)

    return svm, accuracy, training_time

def predict_single_image_svm(model, image, scaler):
    """
    Dự đoán nhãn cho một ảnh đơn lẻ
    """
    # Chuẩn hóa ảnh
    image_scaled = scaler.transform(image.reshape(1, -1))

    # Dự đoán
    prediction = model.predict(image_scaled)

    return prediction[0]
