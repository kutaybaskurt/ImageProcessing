from keras.preprocessing.image import ImageDataGenerator

input_shape = (224, 224, 3)

# Prepare the data generator for testing
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    '/Users/kutaybaskurt/Desktop/Engineering Project/Dataset',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

from keras.saving import load_model
# Load the pre-trained model
model = load_model('/Users/kutaybaskurt/Desktop/Engineering Project/Models And Graphics/xception.h5')

# Evaluate the model on the test data
evaluation = model.evaluate(test_generator)

# Print the evaluation results
print("Evaluation Loss:", evaluation[0])
print("Evaluation Accuracy:", evaluation[1])
