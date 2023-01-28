import os
import numpy as np
from keras.preprocessing import image
from keras.utils import image_utils
import cv2
import dlib
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = './Datasets'
images_dir = os.path.join(basedir,'celeba')
test_images_dir = os.path.join(basedir,'celeba_test')
labels_filename = 'labels.csv'



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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


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
        image_paths = [os.path.join(test_images_dir, "img", img_name) for img_name in labels_df['img_name'].values]
    else:
        print("Getting train images")
        labels_df = pd.read_csv(os.path.join(images_dir, labels_filename), sep='\t')
        image_paths = [os.path.join(images_dir, "img", img_name) for img_name in labels_df['img_name'].values]

    gender_labels = labels_df['gender'].values
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for i, img_path in enumerate(image_paths):
            # load image
            img = image_utils.img_to_array(
                image_utils.load_img(img_path,
                               target_size=None,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[i])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels

def get_training_data():
    X, y = extract_features_labels()
    Y = np.array(y).T
    return X, Y

def get_testing_data():
    X, y = extract_features_labels(test=True)
    Y = np.array(y).T
    return X, Y

def img_SVM_with_cv(training_images, training_labels):
    print("SVM with Cross-Validation")
    classifier = SVC(kernel='linear')
    cv_scores = cross_val_score(classifier, training_images, training_labels, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Score:", np.mean(cv_scores))
    print("Standard Deviation:", np.std(cv_scores))

def img_logistic_regression_with_cv(training_images, training_labels):
    print("Logistic Regression with Cross-Validation")
    classifier = LogisticRegression(max_iter=5000)
    cv_scores = cross_val_score(classifier, training_images, training_labels, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Score:", np.mean(cv_scores))
    print("Standard Deviation:", np.std(cv_scores))



def img_random_forest_with_cv(training_images, training_labels):
    print("Random Forest with Cross-Validation")
    classifier = RandomForestClassifier()
    cv_scores = cross_val_score(classifier, training_images, training_labels, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Score:", np.mean(cv_scores))
    print("Standard Deviation:", np.std(cv_scores))

def img_decision_tree_with_cv(training_images, training_labels):
    print("Decision Tree with Cross-Validation")
    classifier = DecisionTreeClassifier()
    cv_scores = cross_val_score(classifier, training_images, training_labels, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Score:", np.mean(cv_scores))
    print("Standard Deviation:", np.std(cv_scores))

def img_mlp_with_cv(training_images, training_labels):
    print("MLP with Cross-Validation")
    classifier = MLPClassifier()
    cv_scores = cross_val_score(classifier, training_images, training_labels, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Score:", np.mean(cv_scores))
    print("Standard Deviation:", np.std(cv_scores))

def img_SVM_with_randomized_search(training_images, training_labels):
    print("SVM with Randomized Search")
    classifier = SVC(kernel='linear')
    param_distributions = {'C': np.logspace(-3, 3, 7), 'kernel': ['linear']}
    random_search = RandomizedSearchCV(classifier, param_distributions, cv=5, n_iter=10)
    random_search.fit(training_images, training_labels)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

def img_logistic_regression_with_randomized_search(training_images, training_labels):
    print("Logistic Regression with Randomized Search")
    classifier = LogisticRegression(max_iter=5000)
    param_distributions = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
    random_search = RandomizedSearchCV(classifier, param_distributions, cv=5, n_iter=10)
    random_search.fit(training_images, training_labels)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

def img_random_forest_with_randomized_search(training_images, training_labels):
    print("Random Forest with Randomized Search")
    classifier = RandomForestClassifier()
    param_distributions = {'n_estimators': np.arange(10, 100, 10), 'max_depth': np.arange(1, 11)}
    random_search = RandomizedSearchCV(classifier, param_distributions, cv=5, n_iter=10)
    random_search.fit(training_images, training_labels)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

def img_decision_tree_with_randomized_search(training_images, training_labels):
    print("Decision Tree with Randomized Search")
    classifier = DecisionTreeClassifier()
    param_distributions = {'max_depth': np.arange(1, 11)}
    random_search = RandomizedSearchCV(classifier, param_distributions, cv=5, n_iter=10)
    random_search.fit(training_images, training_labels)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

def img_KNN_with_randomized_search(training_images, training_labels):
    print("K-Nearest Neighbors with Randomized Search")
    classifier = KNeighborsClassifier()
    param_distributions = {'n_neighbors': np.arange(1, 30), 'weights': ['uniform', 'distance']}
    random_search = RandomizedSearchCV(classifier, param_distributions, cv=5, n_iter=10)
    random_search.fit(training_images, training_labels)
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

def main():
    tr_X, tr_Y = get_training_data()
    #te_X, te_Y = get_testing_data()

    # Use the new functions that perform cross-validation
    img_SVM_with_cv(tr_X.reshape((tr_X.shape[0], 68*2)), tr_Y)
    print(" ")
    img_SVM_with_randomized_search(tr_X.reshape((tr_X.shape[0], 136)), tr_Y)
    print(" ")
    img_logistic_regression_with_cv(tr_X.reshape((tr_X.shape[0], 68*2)), tr_Y)
    print(" ")
    img_logistic_regression_with_randomized_search(tr_X.reshape((tr_X.shape[0], 136)), tr_Y)
    print(" ")
    img_random_forest_with_cv(tr_X.reshape((tr_X.shape[0], 68*2)), tr_Y)
    print(" ")
    img_random_forest_with_randomized_search(tr_X.reshape((tr_X.shape[0], 136)), tr_Y)
    print(" ")
    img_decision_tree_with_cv(tr_X.reshape((tr_X.shape[0], 68*2)), tr_Y)
    print(" ")
    img_decision_tree_with_randomized_search(tr_X.reshape((tr_X.shape[0], 136)), tr_Y)
    print(" ")
    img_mlp_with_cv(tr_X.reshape((tr_X.shape[0], 68*2)), tr_Y)
    print(" ")
    img_KNN_with_randomized_search(tr_X.reshape((tr_X.shape[0], 136)), tr_Y)

main()