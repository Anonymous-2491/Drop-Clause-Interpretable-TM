import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator

num_classes = 10

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28)).astype("float32")
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    
data_dir = r'./mnist_c'  #Change the data_dir to where you extracted the mnist_c.zip
#Simply comment out the corruptions that you don't want to use. Remember to keep the 'identity' as it is the actual test_images.
CORRUPTIONS = [
    'identity',
    'shot_noise',
    'impulse_noise',
    'glass_blur',
    'motion_blur',
    'shear',
    'scale',
    'rotate',
    'brightness',
    'translate',
    'fog',
    'stripe',
    'spatter',
    'dotted_line',
    'zigzag',
    'canny_edges',
]

_TEST_IMAGES_FILENAME = 'test_images.npy'
_TEST_LABELS_FILENAME = 'test_labels.npy'

# Change esp to change the probability of getting a corrupted image during testing.
esp = 0.5
all_test_images = []
all_test_labels = []

for corruption in CORRUPTIONS:
    images_file = os.path.join(data_dir, corruption, _TEST_IMAGES_FILENAME)
    labels_file = os.path.join(data_dir, corruption, _TEST_LABELS_FILENAME)
    images = np.load(images_file)
    labels = np.load(labels_file)
    all_test_images.append(images)
    all_test_labels.append(labels)

all_test_images = np.array(all_test_images)
all_test_labels = np.array(all_test_labels)

print(all_test_images.shape, all_test_labels.shape)
print("___________________________________________")

X_test = []
Y_test = []

for i in range(10000):
    corrupt = np.random.choice(range(len(CORRUPTIONS)),p=[1-esp]+[esp/(len(CORRUPTIONS)-1)]*(len(CORRUPTIONS)-1))
    X_test.append(all_test_images[corrupt][i])
    Y_test.append(all_test_labels[corrupt][i])
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_test.shape, Y_test.shape)
    
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes)

# Normalize pixel values to be between -1 and 1
train_images, X_test = train_images / 127.5 - 1, X_test / 127.5 - 1

# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

mnist_mlp = tf.keras.models.Sequential([
    # In the first layer we only quantize the weights and not the input
    tf.keras.layers.Flatten(),
    
    lq.layers.QuantDense(500, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantDense(300, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    
    lq.layers.QuantDense(200, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantDense(100, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantDense(10, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
    tf.keras.layers.Activation("softmax")
])

mnist_mlp.compile(
    tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trained_model = mnist_mlp.fit(
    train_images, 
    train_labels,
    batch_size=32, 
    epochs=100,
    validation_data=(X_test, Y_test),
    shuffle=True
)
