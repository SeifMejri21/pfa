from src.data_loader import DataLoader

# images_size = (64, 64)
from src.deep_model import CNNModel

images_size = (100, 100)
dl = DataLoader(images_size=images_size)
labels_name, class_ids = dl.labels_csv_reader()

img_path = "C:/Users/Administrator/Desktop/traffic_dataset/traffic_Data/TEST/004_1_0009_1_j.png"
img = dl.read_and_preprocess_single_image(img_path)
print(img.shape)
num_classes = 58
input_shape = (images_size[1], images_size[0], 3)
cnn = CNNModel(input_shape=input_shape, num_classes=num_classes)
cnn.build_model()

file_path = "weights/best.h5"
cnn.load_weights(file_path)

pred = cnn.predict(img)
print("pred: ", pred, labels_name[pred])

