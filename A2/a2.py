from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

class A2:

    def evaluate_best_model(self,X_test,Y_test):
        self.best_model = self.grid.best_estimator_
        self.label_test_predict = self.best_model.predict(X_test)
        self.best_model_score = accuracy_score(Y_test, self.label_test_predict)
        self.conf_matrix = confusion_matrix(Y_test, self.label_test_predict)
        print("Best Parameters:", self.grid.best_params_)
        print("Best Model Score:", self.best_model_score)
        print("Confusion Matrix:", self.conf_matrix)

    def draw_SVM_learning_curve(self, X_train,Y_train,):
        classifier = SVC(C=self.grid.best_params_['C'], kernel=self.grid.best_params_['kernel'])
        train_sizes, train_scores, val_scores = learning_curve(
                classifier, X_train, Y_train, cv=5, scoring='neg_root_mean_squared_error',train_sizes=np.linspace(0.01, 1.0, 15)
            )
        plt.style.use('seaborn')
        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)
        plt.plot(train_sizes, train_scores_mean, label="Training score")
        plt.plot(train_sizes, val_scores_mean, label="Cross-validation score")
        plt.ylabel('RMSE', fontsize = 14)
        plt.xlabel('Training set size', fontsize = 14)
        plt.title('Learning curve for a SVM model', fontsize = 18, y = 1.03)
        plt.legend()
        plt.show()

    def SVM_with_GridSearchCV(self, X_train,Y_train, C, kernel):
        print("SVM with GridSearchCV")
        param_grid = { "C": C, "kernel": kernel }
        classifier = SVC()
        grid = GridSearchCV(classifier, param_grid, refit=True, verbose=3, cv=5)
        grid.fit(X_train, Y_train)
        self.grid = grid

    def SVM(self, X_train,Y_train, X_test,Y_test, C, kernel):
        print("SVM")
        classifier = SVC(C=C, kernel=kernel)
        classifier.fit(X_train, Y_train)
        pred = classifier.predict(X_test)
        print("Accuracy:", accuracy_score(Y_test, pred))
    
    def train_CNN(self,X_train,Y_train, epochs=10, batch_size=32, learning_rate=0.0001):
        # Define the model architecture
        self.model = Sequential()
        self.model.add(Rescaling(1./255, input_shape=(X_train.shape[1],X_train.shape[2],1)))
        self.model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy(), metrics=['accuracy'])
        self.model.summary()
        Y_train[Y_train == -1] = 0
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1, shuffle=True, use_multiprocessing=True)

    def evaluate_CNN(self,X_test,Y_test):
        Y_test[Y_test == -1] = 0
        test_loss, test_acc = self.model.evaluate(X_test, Y_test, verbose=2)
        print('Test accuracy:', test_acc)
        print("Test Loss:", test_loss)
    
    def draw_CNN_learning_curve(self):
        plt.style.use('seaborn')
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.show()