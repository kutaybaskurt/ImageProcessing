import os
import pickle
import matplotlib.pyplot as plt

# Define the base directory where the .pkl files are located
base_dir = '/Users/kutaybaskurt/Desktop/adsız klasör'

# Define the model names and their corresponding .pkl files
models = {
    'InceptionResNetV2': 'inceptionresnetv2-history.pkl',
    'InceptionV3': 'inceptionv3-history.pkl',
    'ResNet50': 'resnet50-history.pkl',
    'Xception': 'xception-history.pkl'
}

# This function will load the training history from a .pkl file
def load_history(model_name, base_dir):
    history_path = os.path.join(base_dir, models[model_name])
    with open(history_path, 'rb') as file:
        history = pickle.load(file)
    return history

# Iterate over the models and plot their training/validation accuracies and losses
for model_name in models.keys():
    history = load_history(model_name, base_dir)

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
