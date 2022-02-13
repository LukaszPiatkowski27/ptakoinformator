import os

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

image_path = os.path.join(os.path.dirname("dataset"), "dataset")
data = DataLoader.from_folder(image_path)

train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

model = image_classifier.create(train_data, validation_data=validation_data, use_augmentation=True, epochs=10)

loss, accuracy = model.evaluate(test_data)

model.export(export_dir='.')
