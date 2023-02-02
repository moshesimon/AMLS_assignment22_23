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
run_b2 = False

if run_a1 or run_a2:
    train_celeb_images = load_images(images_dir=celeba_images_dir,file_type='jpg', num_imgs=5000) # load images
    test_celeb_images = load_images(images_dir=celeba_test_images_dir,file_type='jpg', num_imgs=1000) # load images
    
    train_gender_celeb_labels = load_labels(label_dir=celeba_labels_dir, label_col_name='gender')
    test_gender_celeb_labels = load_labels(label_dir=celeba_test_labels_dir, label_col_name='gender')
    
    train_smiling_celeb_labels = load_labels(label_dir=celeba_labels_dir, label_col_name='smiling')
    test_smiling_celeb_labels = load_labels(label_dir=celeba_test_labels_dir, label_col_name='smiling')

    #train_celeb_features, train_gender_labels, train_smiling_labels = extract_features_labels(train_celeb_images, train_gender_celeb_labels, train_smiling_celeb_labels) # extract features and labels
    #test_celeb_features, test_gender_labels, test_smiling_labels = extract_features_labels(test_celeb_images, test_gender_celeb_labels, test_smiling_celeb_labels) # extract features and labels

if run_a1:
    print("Running A1")
    a1 = A1()
    a1.SVM_with_GridSearchCV(train_celeb_features, train_gender_labels,C=[0.01, 0.1, 1], kernel=["linear"])
    a1.evaluate_best_model(test_celeb_features, test_gender_labels)
    a1.draw_SVM_learning_curve(train_celeb_features, train_gender_labels)
    
    
if run_a2:
    run_SVM = False
    run_CNN = True
    print("Running A2")
    a2 = A2()
    if run_SVM:
        a2.SVM_with_GridSearchCV(train_celeb_features, train_gender_labels,C=[0.01, 0.1, 1], kernel=["linear"])
        a2.evaluate_best_model(test_celeb_features, test_gender_labels)
        a2.draw_SVM_learning_curve(train_celeb_features, train_gender_labels)
    if run_CNN:
        a2.train_CNN(train_celeb_images,train_smiling_celeb_labels, epochs=8, batch_size=32, learning_rate=0.0001)
        a2.evaluate_CNN(test_celeb_images,test_smiling_celeb_labels)
        a2.draw_CNN_learning_curve()

if run_b1:
    run_SVM_with_features = False
    run_SVM_with_images = True
    run_CNN = False
    train_cartoon_images = load_images(images_dir=cartoon_images_dir,file_type='png', num_imgs=10000) # load images
    test_cartoon_images = load_images(images_dir=cartoon_test_images_dir,file_type='png', num_imgs=2500) # load images

    train_face_labels = load_labels(label_dir=cartoon_labels_dir, label_col_name='face_shape')
    test_face_labels = load_labels(label_dir=cartoon_test_labels_dir, label_col_name='face_shape')

    
    b1 = B1()
    if run_SVM_with_features:
        train_cartoon_features, updated_train_face_labels = extract_features_labels(train_cartoon_images, train_face_labels) # extract features and labels
        test_cartoon_features, updated_test_face_labels = extract_features_labels(test_cartoon_images, test_face_labels) # extract features and labels
        b1.SVM_with_GridSearchCV(train_cartoon_features, updated_train_face_labels,C=[0.01, 0.1, 1], kernel=["linear"])
        b1.evaluate_best_model(test_cartoon_features, updated_test_face_labels)
        b1.draw_SVM_learning_curve(train_cartoon_features, updated_train_face_labels)
    if run_SVM_with_images:
        b1.SVM_with_GridSearchCV(np.reshape(train_cartoon_images,(10000,500*500)), train_face_labels,C=[0.01], kernel=["linear"])
        b1.evaluate_best_model(np.reshape(test_cartoon_images,(2500,500*500)), test_face_labels)
        b1.draw_SVM_learning_curve(train_cartoon_images, train_face_labels)
        #b1.SVM(np.reshape(train_cartoon_images,(10000,500*500)), train_face_labels,np.reshape(test_cartoon_images,(2500,500*500)), test_face_labels)
    if run_CNN:
        b1.train_CNN(train_cartoon_images,train_face_labels, epochs=10, batch_size=32, learning_rate=0.0001)
        b1.evaluate_CNN(test_cartoon_images,test_face_labels)
        b1.draw_CNN_learning_curve()

if run_b2:
    train_cartoon_images_colour = load_images(images_dir=cartoon_images_dir,file_type='png', num_imgs=10000, grayscale=False) # load images
    test_cartoon_images_colour = load_images(images_dir=cartoon_test_images_dir,file_type='png', num_imgs=2500, grayscale=False) # load images
    
    train_cartoon_labels = load_labels(label_dir=cartoon_labels_dir, label_col_name='eye_color')
    test_cartoon_labels = load_labels(label_dir=cartoon_test_labels_dir, label_col_name='eye_color')
    
    x_train, y_train = extract_eyes(train_cartoon_images_colour, train_cartoon_labels)
    x_test, y_test = extract_eyes(test_cartoon_images_colour, test_cartoon_labels)
    b2 = B2(x_train, y_train, x_test, y_test)
    b2.build_model(width=23, height=32, learning_rate=0.001)
    b2.train(epoch=10, batch_size=32)
    b2.evaluate()
    b2.plot()