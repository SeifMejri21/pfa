import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


class MachineLearningModel(object):
    @staticmethod
    def train_pca(images, n_components=50):
        images_vectors = [img.flatten() for img in images]
        pca = PCA(n_components=n_components)
        pca.fit_transform(images_vectors)
        total_variance = sum(pca.explained_variance_ratio_) * 100
        print(f"Total variance retained: {total_variance:.2f}%")
        return pca

    @staticmethod
    def predict_pca(pca, images):
        images_vectors = [img.flatten() for img in images]
        pca_components = pca.fit_transform(images_vectors)
        return pca_components

    @staticmethod
    def save_pca_weights(weights_path):
        1

    @staticmethod
    def load_pca_weights(weights_path):
        weights = 1
        return weights

    @staticmethod
    def train_xgboost_model(pca_features, one_hot_labels):
        labels = np.argmax(one_hot_labels, axis=1)
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(pca_features, labels)
        return model

    @staticmethod
    def predict_with_xgboost(model, pca_features):
        predictions = model.predict(pca_features)
        return predictions

    @staticmethod
    def calculate_xgboost_accuracy(one_hot_labels, predictions):
        true_labels = np.argmax(one_hot_labels, axis=1)
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy

    @staticmethod
    def save_xgboost_weights(weights_path):
        1

    @staticmethod
    def load_xgboost_weights(weights_path):
        weights = 1
        return weights
