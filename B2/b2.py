import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Define the model architecture

class B2:

    def SVM(self, X_train,Y_train, X_test,Y_test, C, kernel):
        print("SVM")
        classifier = SVC(C=C, kernel=kernel)
        classifier.fit(X_train, Y_train)
        pred = classifier.predict(X_test)
        print("Accuracy:", accuracy_score(Y_test, pred))
    
    def train_CNN(self,X_train,Y_train, epochs=10, batch_size=32, learning_rate=0.0001):
        # Define the model architecture
        self.model = Sequential()
        self.model.add(Rescaling(1./255, input_shape=(X_train.shape[1],X_train.shape[2],3)))
        self.model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(5, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy(), metrics=['accuracy'])
        self.model.summary()
        Y_train = tf.one_hot(Y_train, 5)
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1, shuffle=True, use_multiprocessing=True)

    def evaluate_CNN(self,X_test,Y_test):
        Y_test = tf.one_hot(Y_test, 5)
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