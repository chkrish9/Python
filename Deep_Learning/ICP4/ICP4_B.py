# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

K.common.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# load and evaluate a saved model


# load model
model = load_model('updated_new_model.h5')

# summarize model.
model.summary()

# evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

# prediction for the first image

object_list = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse',
               8: 'Ship', 9: 'Truck'}

for i in range(4):
    label = (int(model.predict_classes(X_test[[i], :])))
    plt.imshow((X_test[i]))
    plt.title(object_list[label], fontsize=30)
    plt.show()
