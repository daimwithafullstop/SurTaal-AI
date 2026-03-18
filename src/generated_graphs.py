import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow.keras as keras

DATA_PATH = "data.json"
GENRES = ["Bhangra", "Ghazal", "HipHop", "Pop", "Qawwali"]

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def plot_history(history):
    """Generates the Accuracy & Loss Curves (The 'Learning Curves')"""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    #accuracy curve
    axs[0].plot(history.history["accuracy"], label="Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Model Accuracy over Epochs")
    axs[0].legend(loc="lower right")
    axs[0].grid(True)

    #loss curve
    axs[1].plot(history.history["loss"], label="Train Loss")
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Error (Loss)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Model Error over Epochs")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("graph_learning_curves.png") 
    print("Saved: graph_learning_curves.png")
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """Generates the Heatmap showing where the model gets confused"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=GENRES, yticklabels=GENRES)
    
    plt.xlabel('Predicted Genre')
    plt.ylabel('Actual Genre')
    plt.title('Confusion Matrix (Darker Green = Better)')
    
    plt.savefig("graph_confusion_matrix.png") 
    print("Saved: graph_confusion_matrix.png")
    plt.show()

def main():
  
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        
        keras.layers.Conv2D(32, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("--> Training Model to generate graphs...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30, verbose=1)

    print("\nGENERATING EVALUATION METRICS")
    
    plot_history(history)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    plot_confusion_matrix(y_test, y_pred)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=GENRES))

    with open("evaluation_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, target_names=GENRES))
    print("Saved: evaluation_report.txt")

if __name__ == "__main__":
    main()