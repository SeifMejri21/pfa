from src.data_loader import DataLoader
from src.ml_model import MachineLearningModel

# images_size = (64, 64)
images_size = (100, 100)
dl = DataLoader(images_size=images_size)
ml = MachineLearningModel()

x_train, x_val, y_train, y_val = dl.read_images()
print("images_list: ", len(x_train), x_train.shape)
print("images_list: ", len(x_val), x_val.shape)
print("labels_list: ", len(y_train), y_train.shape)
print("labels_list: ", len(y_val), y_val.shape)


n_components = 80
pca_model = ml.train_pca(x_train, n_components=n_components)
pca_data_train = ml.predict_pca(pca_model, x_train)
pca_data_val = ml.predict_pca(pca_model, x_val)

xgboost_model = ml.train_xgboost_model(pca_data_train, y_train)
predictions = ml.predict_with_xgboost(xgboost_model, pca_data_val)
accuracy = ml.calculate_xgboost_accuracy(pca_data_val, predictions)
print("accuracy: ", accuracy*100)