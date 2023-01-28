from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,GridSearchCV, learning_curve
 
from sklearn.metrics import confusion_matrix
import numpy as np

C = [0.01, 0.1, 1]
gamma = [10, 1, 0.1, 0.01]
kernel = ["linear"]

class A1:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def train_grid_fit(self, model):
        param_grid = {
            "C": C,
            "gamma": gamma,
            "kernel": kernel,
        }

        grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=5)
        grid.fit(self.train_data, self.train_labels)
        self.grid = grid
        self.results = grid.cv_results_

    def evaluate_best_model(self):
        self.best_model = self.grid.best_estimator_.score(
            self.test_data, self.test_labels
        )
        self.label_test_predict = self.grid.predict(self.test_data)
        self.conf_matrix = confusion_matrix(self.test_labels, self.label_test_predict)

    def draw_learning_curve(self):
        train_sizes, train_scores, val_scores = learning_curve(
                self.grid.best_estimator_, self.train_data, self.train_labels, cv=5
            )

    def SVM_with_GridSearchCV(self):
        print("SVM with GridSearchCV")
        classifier = SVC()
        self.train_grid_fit(classifier)
        self.evaluate_best_model()
        print("Best Parameters:", self.grid.best_params_)
        print("Best Model Score:", self.best_model)
        print("Confusion Matrix:", self.conf_matrix)

    def SVM_with_cv(self, training_images, training_labels):
        print("SVM with Cross-Validation")
        classifier = SVC(kernel='linear')
        cv_scores = cross_val_score(classifier, training_images, training_labels, cv=5)
        print("Cross-Validation Scores:", cv_scores)
        print("Mean Score:", np.mean(cv_scores))
        print("Standard Deviation:", np.std(cv_scores))