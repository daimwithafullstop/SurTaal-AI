import json
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data.json"
MODEL_PATH = "pakistani_music_model.h5"
GENRES = ["Bhangra", "Ghazal", "HipHop", "Pop", "Qawwali"]

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def plot_confusion_matrix(y_test, y_pred):
    """Generates the Heatmap for Question 5"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=GENRES, yticklabels=GENRES)
    
    plt.xlabel('Predicted Genre')
    plt.ylabel('Actual Genre')
    plt.title('Confusion Matrix')
    
    print("--> Saving Confusion Matrix image...")
    plt.savefig("confusion_matrix.png")
    plt.show()

def main():

    print("Loading data...")
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    X_test = X_test[..., np.newaxis]

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = keras.models.load_model(MODEL_PATH)
    except IOError:
        print(f"ERROR: Could not find '{MODEL_PATH}'. Did you run the training script first?")
        return

    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(y_test, y_pred, target_names=GENRES)
    print(report)

    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    
    try:
        import seaborn
        main()
    except ImportError:
        print("You need to install seaborn for the graph.")
        print("Run this in terminal: pip install seaborn")