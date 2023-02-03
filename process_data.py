import os
import dlib
import cv2
import numpy as np
import pandas as pd
from keras.utils import image_utils

predictor_dir = os.path.join(os.path.abspath(''),'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_dir)


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
    # to a 2-tuple of (x, y)-coordiates
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
    
    # detect faces in the grayscale image
    rects = detector(image, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(image, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout

def load_labels(label_dir, label_col_name):
    labels_df = pd.read_csv(label_dir, sep='\t')
    labels = labels_df[label_col_name].values
    labels = np.array(labels)
    return labels

def load_images(images_dir, file_type, num_imgs, grayscale=True):
    
    image_paths = [os.path.join(images_dir, f"{img_num}.{file_type}") for img_num in range(num_imgs)]
    
    all_images = []
    if os.path.isdir(images_dir):
        
        for img_path in image_paths:
            # load image
            img = image_utils.load_img(img_path, target_size=None, interpolation='bicubic')
            img_array = image_utils.img_to_array(img)
            if grayscale:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array = img_array.astype('uint8')
            all_images.append(img_array)


    images = np.array(all_images)

    return images

def extract_features_labels(images, labels_1, labels_2 = False):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    all_features = []

    for i, img in enumerate(images):
        features = run_dlib_shape(img)
        if features is not None:
            all_features.append(features)
        else:
            labels_1 = np.delete(labels_1, i, axis=0)
            labels_1 = np.insert(labels_1,0,-5,axis=0) # insert -5 to the first row to the indexes constant
            if labels_2 is not False:
                labels_2 = np.delete(labels_2, i, axis=0) 
                labels_2 = np.insert(labels_2,0,-5,axis=0) # insert -5 to the first row to the indexes constant

    labels_1 = labels_1[labels_1 != -5]
    if labels_2 is not False:
        labels_2 = labels_2[labels_2 != -5]
    features = np.array(all_features)
    features = features.reshape((features.shape[0], 68*2))
    return features, labels_1, labels_2

def extract_eyes(images, eye_labels):
    all_eyes = []
    for i, img in enumerate(images):
        if has_sunglasses(img):
            eye_labels = np.delete(eye_labels, i, axis=0)
            eye_labels = np.insert(eye_labels,0,-5,axis=0)
            continue
        eye = img[250:273,190:222]
        all_eyes.append(eye)
    eyes = np.array(all_eyes)
    eye_labels = eye_labels[eye_labels != -5]
    return eyes, eye_labels

def has_sunglasses(image):
    #check if the person is wearing sunglasses
    white_of_the_eye = image[265:280,218:222]
    x = white_of_the_eye.flatten()
    if np.average(x) < 200:
        return True
    return False