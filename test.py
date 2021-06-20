import keras
from sklearn.metrics import classification_report
from utils import wav2mfcc, model, get_data
from keras.utils import to_categorical

def check_preds(X, y):

    trained_model = keras.models.load_model('my_model.h5')
    test_metrics = trained_model.evaluate(X, y,verbose = 1, return_dict = True)

    print (test_metrics)
    # predictions = trained_model.predict(X)
    
    # labels = predictions.argmax(axis=-1)

    # print(y[0:20])
    # print(labels[0:20])



    # print(classification_report(y, to_categorical(predictions)))


