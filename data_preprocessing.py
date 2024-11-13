# data_preprocessing.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_dir, image_size=(64, 64)):
    """
    Load and preprocess images from the given directory with progress bar and error handling
    """
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")

    # Get total number of files for progress bar
    total_files = sum([len(files) for _, _, files in os.walk(data_dir)])

    if total_files == 0:
        raise ValueError(f"No files found in directory {data_dir}")

    print("Loading and preprocessing images...")
    pbar = tqdm(total=total_files, desc="Processing images")

    # Đọc tất cả các thư mục con
    for class_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(folder_path):
            label_dict[current_label] = class_folder

            # Đọc tất cả các ảnh trong thư mục
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                try:
                    # Đọc và kiểm tra ảnh
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"\nWarning: Could not read image {image_path}")
                        continue

                    # Xử lý ảnh
                    img = cv2.resize(img, image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    img_flat = img.flatten()

                    images.append(img_flat)
                    labels.append(current_label)

                except Exception as e:
                    print(f"\nError processing image {image_path}: {str(e)}")
                finally:
                    pbar.update(1)

            current_label += 1

    pbar.close()

    if not images:
        raise ValueError("No valid images were loaded")

    X = np.array(images)
    y = np.array(labels)

    print(f"\nTotal images loaded: {len(images)}")
    print(f"Number of classes: {current_label}")

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, label_dict, scaler


def evaluate_model(model_name, y_true, y_pred, training_time):
    """
    Đánh giá model với xử lý lỗi
    """
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        accuracy = accuracy_score(y_true, y_pred)

        print(f"\nKết quả đánh giá cho {model_name}:")
        print(f"Độ chính xác: {accuracy * 100:.2f}%")
        print(f"Thời gian training: {training_time:.2f} giây")

        print("\nBáo cáo phân loại chi tiết:")
        print(classification_report(y_true, y_pred))

        # Vẽ confusion matrix với seaborn
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="white")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return accuracy, training_time

    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        return None, training_time