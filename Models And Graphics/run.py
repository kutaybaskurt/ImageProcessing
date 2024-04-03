from keras.applications import ResNet50, InceptionResNetV2, InceptionV3, EfficientNetB7, EfficientNetV2L, Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# Define the input shape and base model
input_shape = (224, 224, 3)

bms = {
    # Uncomment the models you want to use
    # 'resnet50': ResNet50(weights='imagenet', include_top=False, input_shape=input_shape),
    # 'inceptionresnetv2': InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
    # 'inceptionv3': InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape),
    'xception': Xception(weights='imagenet', include_top=False, input_shape=input_shape),
}

for key in bms:
  base_model = bms[key]
  # Add custom layers on top of the base model
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(5, activation='softmax')(x)

  # Create the model
  model = Model(inputs=base_model.input, outputs=predictions)

  # Make the layers of the pre-trained model trainable
  for layer in base_model.layers:
      layer.trainable = True

  # Compile the model
  model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

  # Print the model summary
  model.summary()

  # Data augmentation and loading
  train_datagen = ImageDataGenerator(
      rescale=1./255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  test_datagen = ImageDataGenerator(rescale=1./255)

  batch_size = 32

  # Prepare training data
  train_generator = train_datagen.flow_from_directory(
      '/Users/kutaybaskurt/Desktop/Engineering Project/Dataset',
      target_size=input_shape[:2],
      batch_size=batch_size,
      class_mode='categorical'
  )

  # Prepare validation data
  validation_generator = test_datagen.flow_from_directory(
      '/Users/kutaybaskurt/Desktop/Engineering Project/Dataset',
      target_size=input_shape[:2],
      batch_size=batch_size,
      class_mode='categorical'
  )

  # Train the model
  history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
  )

  # Save the model and training history
  import pickle
  model_name = key
  model.save('/Users/kutaybaskurt/Desktop/Engineering Project/Models And Graphics/'+model_name+'.h5')
  with open('/Users/kutaybaskurt/Desktop/Engineering Project/Models And Graphics/history_'+model_name+'.pkl','wb') as f:
    pickle.dump(history.history, f)
