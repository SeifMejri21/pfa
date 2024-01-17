from src.data_loader import DataLoader
from src.ml_model import MachineLearningModel

# images_size = (64, 64)
from src.utils.deep_model import CNNModel

images_size = (100, 100)
dl = DataLoader(images_size=images_size)
ml = MachineLearningModel()

x_train, x_val, y_train, y_val = dl.read_images()
print("images_list: ", len(x_train), x_train.shape)
print("images_list: ", len(x_val), x_val.shape)
print("labels_list: ", len(y_train), y_train.shape)
print("labels_list: ", len(y_val), y_val.shape)

# num_classes = y_train.shape[1]
# input_shape = (images_size[1], images_size[0], 3)
# cnn = CNNModel(input_shape=input_shape, num_classes=num_classes)
# cnn.build_model()
# cnn.model.summary()
# cnn.train(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
# # predictions = cnn.predict(test_images)
