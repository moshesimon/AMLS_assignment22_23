import tensorflow as tf
import numpy as np

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

class B1:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels 

    # Cross-Entropy Loss.
    # Note that this will apply 'softmax' to the logits.
    def cross_entropy_loss(self, x, y):
        # Convert labels to int 64 for tf cross-entropy function.
        y = tf.cast(y, tf.int64)
        # Apply softmax to logits and compute cross-entropy.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        # Average loss across the batch.
        return tf.reduce_mean(loss)

    # Accuracy metric.
    def accuracy(self, y_pred, y_true):
        # Predicted class is the index of highest score in prediction vector (i.e. argmax).
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    # Stochastic gradient descent optimizer.
    

    # Optimization process. 
    def run_optimization(self, x, y):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass.
            pred = self.conv_net(x, is_training=True)
            # Compute loss.
            loss = self.cross_entropy_loss(pred, y)
            
        # Variables to update, i.e. trainable variables.
        trainable_variables = self.conv_net.trainable_variables

        # Compute gradients.
        gradients = g.gradient(loss, trainable_variables)
        
        # Update W and b following gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def train_CNN(self):
        # Build neural network model.
        self.conv_net = ConvNet()
        self.optimizer = tf.optimizers.Adam(learning_rate)

        x_train, x_test = np.array(self.train_data, np.float32), np.array(self.test_data, np.float32)
        # Normalize images value from [0, 255] to [0, 1].
        x_train, x_test = x_train / 255., x_test / 255.

        # Use tf.data API to shuffle and batch data.
        train_data = tf.data.Dataset.from_tensor_slices((x_train, self.train_labels))
        train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

    # Run training for the given number of steps.
        for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
            # Run the optimization to update W and b values.
            self.run_optimization(batch_x, batch_y)
            print(step)
            if step % display_step == 0:
                pred = self.conv_net(batch_x)
                loss = self.cross_entropy_loss(pred, batch_y)
                acc = self.accuracy(pred, batch_y)
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

        # Test model on validation set.
        pred = self.conv_net(x_test)
        print("Test Accuracy: %f" % self.accuracy(pred, self.test_labels))