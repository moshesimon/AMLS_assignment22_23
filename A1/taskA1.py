import os
import numpy as np
from keras.preprocessing import image
from keras.utils import image_utils
import cv2
import dlib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './Datasets'
images_dir = os.path.join(basedir,'celeba')
test_images_dir = os.path.join(basedir,'celeba_test')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(test=False):
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
        image_paths = [os.path.join(test_images_dir, "img", l) for l in os.listdir(os.path.join(test_images_dir, "img"))]
        target_size = None
        labels_file = open(os.path.join(test_images_dir, labels_filename), 'r')
    else:
        print("Getting train images")
        image_paths = [os.path.join(images_dir, "img", l) for l in os.listdir(os.path.join(images_dir, "img"))]
        target_size = None
        labels_file = open(os.path.join(images_dir, labels_filename), 'r')

    lines = labels_file.readlines()
    gender_labels = {line.strip().split('\t')[0] : int(line.strip().split('\t')[2]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name= img_path.split('.')[1].split('\\')[-1]

            # load image
            img = image_utils.img_to_array(
                image_utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[file_name])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels

def get_training_data():
    X, y = extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    return X, Y

def get_testing_data():
    X, y = extract_features_labels(test=True)
    Y = np.array([y, -(y - 1)]).T
    return X, Y

def allocate_weights_and_biases():

    # define number of hidden layers ..
    n_hidden_1 = 2048  # 1st layer number of neurons
    n_hidden_2 = 2048  # 2nd layer number of neurons

    # inputs placeholders
    X = tf.placeholder("float", [None, 68, 2])
    Y = tf.placeholder("float", [None, 2])  # 2 output classes
    
    # flatten image features into one vector (i.e. reshape image feature matrix into a vector)
    # images_flat = tf.contrib.layers.flatten(X)  
    images_flat = tf.compat.v1.layers.flatten(X)  
    
    
    # weights and biases are initialized from a normal distribution with a specified standard devation stddev
    stddev = 0.01
    
    # define placeholders for weights and biases in the graph
    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, n_hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, 2], stddev=stddev))
    }

    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
    }
    
    return weights, biases, X, Y, images_flat

# Create model
def multilayer_perceptron():
        
    weights, biases, X, Y, images_flat = allocate_weights_and_biases()

    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.sigmoid(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.sigmoid(layer_2)
    
    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer, X, Y

# learning parameters
learning_rate = 5e-5
training_epochs = 1000

# display training accuracy every ..
display_accuracy_step = 10

    
training_images, training_labels = get_training_data() 
test_images, test_labels = get_testing_data()
logits, X, Y = multilayer_perceptron()

# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# define training graph operation
train_op = optimizer.minimize(loss_op)

# graph operation to initialize all variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:

        # run graph weights/biases initialization op
        sess.run(init_op)
        # begin training loop ..
        for epoch in range(training_epochs):
            # complete code below
            # run optimization operation (backprop) and cost operation (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images,
                                                               Y: training_labels})

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))
                
            if epoch % display_accuracy_step == 0:
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

                # calculate training accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy: {:.3f}".format(accuracy.eval({X: training_images, Y: training_labels})))

        print("Optimization Finished!")

        # -- Define and run test operation -- #
        
        # apply softmax to output logits
        pred = tf.nn.softmax(logits)
        
        #  derive inffered calasses as the class with the top value in the output density function
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        
        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
        # complete code below
        # run test accuracy operation ..
        print("Test Accuracy:", accuracy.eval({X: test_images, Y: test_labels}))