import os
from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint


# Define the path to your dataset
# dataset_path = 'path/to/your/img/folder'
dataset_path="/Users/kutaybaskurt/Desktop/Bitirme/Dataset"
num_classes = len(os.listdir(dataset_path))
print("num_classes",num_classes)
# Set up data generators for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    channel_shift_range=50.0,
    fill_mode='nearest',
    validation_split=0.2
    )
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',  # for multiple categories
    subset='training'
)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_generator = validation_datagen.flow_from_directory(
    dataset_path,
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',  # for multiple categories
    subset='validation'
)

# Build a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))  # num_classes is the number of categories

print("num_classes:", num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
checkpoint = ModelCheckpoint(
    "model_checkpoint.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

# Train the model
model.fit(train_generator, epochs=300, validation_data=validation_generator,callbacks=[checkpoint])

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc}')
