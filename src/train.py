import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
 
DATA_PATH = "data/data.json"

def load_data(data_path):
    """Loads training dataset from json file"""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    # Load the genre mapping 
    mapping = data["mapping"]
    
    return X, y, mapping

def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets"""
    
    # load data
    X, y, mapping = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets (for CNN)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, mapping

def build_model(input_shape):
    """Generates CNN model"""

    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer 
    model.add(keras.layers.Dense(5, activation='softmax')) 

    return model

def predict(model, X, y):
    """Predict a single sample from the test set"""
    
    # Add a dimension 
    X = X[np.newaxis, ...] 

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print(f"Target: {y}, Predicted label: {predicted_index}")


if __name__ == "__main__":

    # 1. create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, mapping = prepare_datasets(0.25, 0.2)

    # 2. build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # 3. compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 4. train the network
    print("\n--> STARTING TRAINING...")
    history = model.fit(X_train, y_train, 
                        validation_data=(X_validation, y_validation), 
                        batch_size=32, 
                        epochs=30) 

    # 5. evaluate the network on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nAccuracy on test set is: {test_accuracy * 100:.2f}%")

    # 6. save the trained model
    model.save("pakistani_music_model.h5")
    print("Model saved to disk as 'pakistani_music_model.h5'")
    
    # 7. Print the mapping so we know which number is which genre
    print("\nGenre Mapping:")
    for i, genre in enumerate(mapping):
        print(f"{i}: {genre}")