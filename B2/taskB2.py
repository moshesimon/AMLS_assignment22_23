from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import os
import cv2
from keras.utils import image_utils
import pandas as pd
#from tensorflow.keras import Model, layers

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './Datasets'
images_dir = os.path.join(basedir,'cartoon_set')
test_images_dir = os.path.join(basedir,'cartoon_set_test')
labels_filename = 'labels.csv'

# MNIST dataset parameters.
num_classes = 5 # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.001
training_steps = 78
batch_size = 128
display_step = 10

# Network parameters.
conv1_filters = 32 # number of filters for 1st conv layer.
conv2_filters = 64 # number of filters for 2nd conv layer.
fc1_units = 1024 # number of neurons for 1st fully-connected layer.

def extract_features_labels(test = False):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    if test:
        print("Getting test images")
        labels_df = pd.read_csv(os.path.join(test_images_dir, labels_filename), sep='\t')
        image_paths = [os.path.join(test_images_dir, "img", img_name) for img_name in labels_df['file_name'].values]
    else:
        print("Getting train images")
        labels_df = pd.read_csv(os.path.join(images_dir, labels_filename), sep='\t')
        image_paths = [os.path.join(images_dir, "img", img_name) for img_name in labels_df['file_name'].values]

    face_shape_labels = labels_df['eye_color'].values
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for i, img_path in enumerate(image_paths):
            # load image
            if i == 500 and test:
                break
            img = image_utils.img_to_array(
                image_utils.load_img(img_path,
                            target_size=None,
                            interpolation='bicubic'))
            resized_image = img.astype('uint8')
            #gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            #gray = gray.astype('uint8')
            all_features.append(resized_image)
            all_labels.append(face_shape_labels[i])

    features = np.array(all_features)
    labels = np.array(all_labels)
    return features, labels

def get_training_data():
    X, Y = extract_features_labels()
    return X, Y

def get_testing_data():
    X, Y = extract_features_labels(test=True)
    return X, Y

x_train, y_train = get_training_data()
x_test, y_test = get_testing_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Create TF Model.
class ConvNet(tf.keras.Model):
    # Set layers.
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = tf.keras.layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = tf.keras.layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layer.
        self.fc1 = tf.keras.layers.Dense(1024)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = tf.keras.layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 500, 500, 3])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# Build neural network model.
conv_net = ConvNet()

# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = conv_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    print(step)
    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test model on validation set.
pred = conv_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))