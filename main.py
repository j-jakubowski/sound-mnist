from comet_ml import Experiment
# experiment = Experiment(project_name="soundmnist")
from tensorflow.python.keras import callbacks
from utils import model, get_data
import test



# X_train, X_test, y_train, y_test, cnn_model = get_data.get_all()
X_train, y_train, X_valid, y_valid, X_test, y_test = get_data.getData()


dim_1 = X_train.shape[1]
dim_2 = X_train.shape[2]
channels = X_train.shape[3]
noOfClasses = 10

input_shape = (dim_1, dim_2, channels)

cnn_model = model.get_cnn_model(input_shape, noOfClasses)

# callback = callbacks.EarlyStopping(monitor='accuracy', patience=50,min_delta=0.01)

print("Training set len: " + str(X_train.shape[0]))

# cnn_model.fit(X_train, y_train, epochs=1000, verbose=1, shuffle = True, validation_data= (X_valid, y_valid))#, callbacks = [callback] )#
#cnn_model.save('my_model.h5')
test.check_preds(X_test, y_test)
