from data_preprocessing import load_and_preprocess_data
from knn_classifier import train_knn
from svm_classifier import train_svm
from ann_classifier import train_ann
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def plot_results(models, accuracies, training_times):
    """
    Vẽ biểu đồ kết quả với style đẹp hơn và xử lý lỗi
    """
    # Set style cho seaborn
    sns.set_theme(style="whitegrid")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot accuracy
    sns.barplot(x=models, y=accuracies, ax=ax1, palette="husl")
    ax1.set_title('Model Accuracy Comparison', pad=20)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)

    # Add percentage labels on bars
    for i, v in enumerate(accuracies):
        ax1.text(i, v, f'{v:.1%}', ha='center', va='bottom')

    # Plot training time
    sns.barplot(x=models, y=training_times, ax=ax2, palette="husl")
    ax2.set_title('Training Time Comparison', pad=20)
    ax2.set_ylabel('Time (seconds)')

    # Add time labels on bars
    for i, v in enumerate(training_times):
        ax2.text(i, v, f'{v:.1f}s', ha='center', va='bottom')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def main():
    try:
        # Đường dẫn đến thư mục chứa dữ liệu
        data_dir = "archive"

        # Load và tiền xử lý dữ liệu
        X_train, X_test, y_train, y_test, label_dict, scaler = load_and_preprocess_data(data_dir)

        # Training và đánh giá các model
        knn_model, knn_accuracy, knn_time = train_knn(X_train, X_test, y_train, y_test)
        svm_model, svm_accuracy, svm_time = train_svm(X_train, X_test, y_train, y_test)
        ann_model, ann_accuracy, ann_time = train_ann(X_train, X_test, y_train, y_test)

        # Vẽ biểu đồ so sánh
        models = ['KNN', 'SVM', 'ANN']
        accuracies = [knn_accuracy, svm_accuracy, ann_accuracy]
        training_times = [knn_time, svm_time, ann_time]

        plot_results(models, accuracies, training_times)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()