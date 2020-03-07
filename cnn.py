# Convolution neural network for Gender and Emotion classification

# importing the packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# making the object of Sequential
classifier_1 = Sequential();

# adding the convolution layer
classifier_1.add(Convolution2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 3)))

# adding the pooling layer
classifier_1.add(MaxPooling2D(pool_size = (2, 2)))

# adding the second convolution and pooling layers
classifier_1.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier_1.add(MaxPooling2D(pool_size = (2, 2)))

# applying flattening
classifier_1.add(Flatten())

# building ANN
classifier_1.add(Dense(units = 128, activation = 'relu'))
classifier_1.add(Dense(units = 1, activation = 'sigmoid'))
classifier_1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the dataset to the CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('dataset_1/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
classifier_1.fit(training_set, epochs = 25, steps_per_epoch = 3300)

# saving the model to disk and loading it again
from keras.models import load_model
classifier_1.save('gender_model.h5')
del classifier_1
classifier_1 = load_model('gender_model.h5')

# making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(path = 'dataset_1/single_prediction/men_or_women_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
training_set.class_indices
result = classifier_1.predict(test_image)
if result[0][0] == 0:
     pred = 'men'
else:
     pred = 'women'

# making another classifier for emotions

# making the object of Sequential
classifier_2 = Sequential();

# adding the convolution layer
classifier_2.add(Convolution2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 3)))

# adding the pooling layer
classifier_2.add(MaxPooling2D(pool_size = (2, 2)))

# adding the second convolution and pooling layers
classifier_2.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier_2.add(MaxPooling2D(pool_size = (2, 2)))

# applying flattening
classifier_2.add(Flatten())

# building ANN
classifier_2.add(Dense(units = 128, activation = 'relu'))
classifier_2.add(Dense(units = 7, activation = 'softmax'))
classifier_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# fitting the dataset to the CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset_2/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('dataset_2/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')
classifier_2.fit_generator(training_set,
                           steps_per_epoch=28000,
                           epochs=25,
                           validation_data=test_set,
                           validation_steps=7000)

# saving the model to disk and loading it again
from keras.models import load_model
classifier_2.save('emotion_model.h5')
del classifier_2
classifier_2 = load_model('emotion_model.h5')

# making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(path = 'dataset_2/single_prediction/emotion_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
training_set.class_indices
result = classifier_2.predict(test_image)
if result[0][0] == 0:
     pred = 'angry'
elif result[0][0] == 1:
     pred = 'disgust'
elif result[0][0] == 2:
     pred = 'fear'
elif result[0][0] == 3:
     pred = 'happy'
elif result[0][0] == 4:
     pred = 'neutral'
elif result[0][0] == 5:
     pred = 'sad'
elif result[0][0] == 6:
     pred = 'surprise'