import matplotlib.pyplot as plt
import pickle
import os

for pkl_file_name in os.listdir('.'):
  if ('.pkl' in pkl_file_name):
    with open(pkl_file_name, 'rb') as f:
      history = pickle.load(f)

      accuracy = history['accuracy']
      val_accuracy = history['val_accuracy']
      loss = history['loss']
      val_loss = history['val_loss']

      # Plotting accuracy
      plt.plot(accuracy, label='Training Accuracy')
      plt.plot(val_accuracy, label='Validation Accuracy')
      plt.title('Training and Validation Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.savefig(pkl_file_name.split('.')[0] + '_acc.png')
      plt.close()

      # Plotting loss
      plt.plot(loss, label='Training Loss')
      plt.plot(val_loss, label='Validation Loss')
      plt.title('Training and Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(pkl_file_name.split('.')[0] + '_loss.png')
      plt.close()
