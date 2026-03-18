import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Reshape, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# --- CONSTANTS ---
DATA_PATH = "data.json"
GENRES = ["Bhangra", "Ghazal", "HipHop", "Pop", "Qawwali"]

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def build_hybrid_model(input_shape):
    model = keras.Sequential()
    
    # CNN Part
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization())

    # Reshape for RNN
    model.add(Reshape((33, -1))) 

    # RNN Part
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.3))

    # Classifier
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    return model

def plot_learning_curves(history):
    """Generates Accuracy and Loss Graphs"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Accuracy Plot
    axs[0].plot(history.history["accuracy"], label="Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Model Accuracy (Learning Curve)")
    axs[0].legend(loc="lower right")
    axs[0].grid(True)

    # Loss Plot
    axs[1].plot(history.history["loss"], label="Train Loss")
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Error (Loss)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Model Error (Loss Curve)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("graph_learning_curves.png")
    print("Saved: graph_learning_curves.png")
    plt.show()

def plot_per_class_metrics(y_true, y_pred):
    """Generates Bar Charts for Precision, Recall, and F1 per Genre"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    x = np.arange(len(GENRES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score')

    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics by Genre')
    ax.set_xticks(x)
    ax.set_xticklabels(GENRES)
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    plt.savefig("graph_per_class_metrics.png")
    print("Saved: graph_per_class_metrics.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Generates Heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GENRES, yticklabels=GENRES)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Genre')
    plt.xlabel('Predicted Genre')
    plt.savefig("graph_confusion_matrix.png")
    print("Saved: graph_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    # 1. Prepare Data
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # 2. Build & Train Hybrid Model
    model = build_hybrid_model((X_train.shape[1], X_train.shape[2], 1))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Starting Training (This generates the history for graphs)...")
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # 3. Predict for Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Generate All Graphs
    print("\n--- Generating Graphs ---")
    plot_learning_curves(history)       # Accuracy & Loss Line Charts
    plot_confusion_matrix(y_test, y_pred) # Heatmap
    plot_per_class_metrics(y_test, y_pred) # Precision/Recall/F1 Bar Charts

    print("\nDone! Check your folder for 3 PNG files.")