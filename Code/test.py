import numpy as np
import cv2
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("Traffic.h5")

# List of class labels corresponding to the model's output
class_labels = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles > 3.5 tons',
    'Right-of-way at intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles > 3.5 tons prohibited', 'No entry', 'General caution', 'Dangerous curve left',
    'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of speed + passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
    'Roundabout mandatory', 'End of no passing', 'End no passing for vehicles > 3.5 tons'
]

def predict_traffic_sign(image_path):
    try:
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            return [{"ERROR": "Invalid image path or unreadable file."}]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (30, 30))  # Resize to 30x30
        image = image.astype('float32') / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)  # Get the class with the highest probability

        # Return the predicted label
        return [{"image": class_labels[predicted_class]}]

    except Exception as e:
        return [{"ERROR": str(e)}]

# Testing the function
if __name__ == "__main__":
    # Specify the path to your test image here
    image_path = r"C:\Users\satwi\Downloads\Road Sign detection\5.jpg"

    # Run prediction
    result = predict_traffic_sign(image_path)

    # Print the result
    if "ERROR" in result[0]:
        print(f"Error: {result[0]['ERROR']}")
    else:
        print(f"Prediction: {result[0]['image']}")
