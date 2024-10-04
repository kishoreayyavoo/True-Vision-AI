import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def load_data(test_dir, batch_size=32, img_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  
    )
    return test_data

def evaluate_model(model, test_data):
    results = model.evaluate(test_data)
    metrics = {
        "loss": results[0],
        "accuracy": results[1]
    }
    with open('evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")

def main():
    model_path = 'D:/deepfake_detection_project/models/saved_models/deepfake_model.h5'
    test_dir = 'D:/deepfake_detection_project/data/processed_frames/test'  # Update this path if using test data
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    test_data = load_data(test_dir)
    evaluate_model(model, test_data)
if __name__ == "__main__":
    main()
