from process_data import * # import all functions from process_data.py
from A1.a1 import A1
from A2.a2 import A2
from B1.b1 import B1
from B2.b2 import B2
# PATH TO ALL IMAGES


basedir = os.path.abspath('')
dataset_dir = os.path.join(basedir,'Datasets')
celeba_images_dir = os.path.join(dataset_dir,'celeba', 'img')
celeba_test_images_dir = os.path.join(dataset_dir,'celeba_test', 'img')
cartoon_images_dir = os.path.join(dataset_dir,'cartoon_set', 'img')
cartoon_test_images_dir = os.path.join(dataset_dir,'cartoon_set_test', 'img')
celeba_labels_dir = os.path.join(dataset_dir,'celeba', 'labels.csv')
celeba_test_labels_dir = os.path.join(dataset_dir,'celeba_test', 'labels.csv')
cartoon_labels_dir = os.path.join(dataset_dir,'cartoon_set', 'labels.csv')
cartoon_test_labels_dir = os.path.join(dataset_dir,'cartoon_set_test', 'labels.csv')

run_a1 = False
run_a2 = False
run_b1 = True
run_b2 = True

if run_a1 or run_a2:
    train_celeb_images = load_images(images_dir=celeba_images_dir,file_type='jpg', num_imgs=5000) # load images
    test_celeb_images = load_images(images_dir=celeba_test_images_dir,file_type='jpg', num_imgs=1000) # load images
    
    train_gender_celeb_labels = load_labels(label_dir=celeba_labels_dir, label_col_name='gender')
    test_gender_celeb_labels = load_labels(label_dir=celeba_test_labels_dir, label_col_name='gender')
    
    train_smiling_celeb_labels = load_labels(label_dir=celeba_labels_dir, label_col_name='smiling')
    test_smiling_celeb_labels = load_labels(label_dir=celeba_test_labels_dir, label_col_name='smiling')

    train_celeb_features, train_gender_labels, train_smiling_labels = extract_features_labels(train_celeb_images, train_gender_celeb_labels, train_smiling_celeb_labels) # extract features and labels
    test_celeb_features, test_gender_labels, test_smiling_labels = extract_features_labels(test_celeb_images, test_gender_celeb_labels, test_smiling_celeb_labels) # extract features and labels

if run_a1:
    print("Running A1")
    a1 = A1(train_celeb_features, train_gender_labels, test_celeb_features, test_gender_labels)
    a1.SVM_with_GridSearchCV()
    #a1.SVM_with_cv(joined_celeb_features, joined_gender_labels)
    
    
if run_a2:
    print("Running A2")
    a2 = A2()
    #a2.SVM_with_cv(features, smiling_labels)

if run_b1 or run_b2:
    train_cartoon_images = load_images(images_dir=cartoon_images_dir,file_type='png', num_imgs=5000) # load images
    test_cartoon_images = load_images(images_dir=cartoon_test_images_dir,file_type='png', num_imgs=1000) # load images
    
    train_cartoon_labels = load_labels(label_dir=cartoon_labels_dir, label_col_name='eye_color')
    test_cartoon_labels = load_labels(label_dir=cartoon_test_labels_dir, label_col_name='eye_color')

if run_b1:
    b1 = B1(train_cartoon_images, train_cartoon_labels, test_cartoon_images, test_cartoon_labels)
    b1.train_CNN()