import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_model(model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    return model

def load_test_data(test_dir, batch_size=32, img_size=(224, 224)):
    # Create an ImageDataGenerator for rescaling
    datagen = ImageDataGenerator(rescale=1./255)

    # Load the test data
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important: do not shuffle test data for evaluation
    )

    return test_data

def evaluate_model(model, test_data):
    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss, test_accuracy

def main():
    model_path = 'D:/deepfake_detection_project/models/saved_models/deepfake_model.h5'  # Path to your trained model
    test_dir = 'D:/deepfake_detection_project/data/processed_frames/test'  # Directory containing test images

    # Load model and test data
    model = load_model(model_path)
    test_data = load_test_data(test_dir)

    # Evaluate the model
    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
