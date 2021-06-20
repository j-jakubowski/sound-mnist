import os

from scipy.signal import wavelets
from utils.wav2mfcc import wav2mfcc
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa import display

def addNoise(data):
    noiseFactor = 0.01
    noise = np.random.randn(len(data))
    augmentedData = data + noise * noiseFactor

    return augmentedData

def decreaseAmplitube(data):
    scalingFactor = np.random.uniform()
    augmentedData = data * scalingFactor
    
    return augmentedData

def increaseAmplitube(data):

    maxVal = max(abs(data))

    scalingFactor =  np.random.uniform(low = maxVal, high = 1.0)
    augmentedData = data / scalingFactor 
    
    return augmentedData
    

def shiftInTime():
    pass

def getDataSet(RecordingsPath, fileList):
    labels = []
    mfccs = []

    for f in fileList:
        if f.endswith('.wav'):

            wave, sr = librosa.load(RecordingsPath + f, mono=True, sr=None)
            label = f.split('_')[0]

            mfccs.append(wav2mfcc(wave, sr))
            labels.append(label)

    mfccs = np.asarray(mfccs)
    mfccs = mfccs.reshape((mfccs.shape[0], mfccs.shape[1], mfccs.shape[2], 1))

    return mfccs, to_categorical(labels)

def getTrainingSet(RecordingsPath, fileList, augmentDataSet = True):
    labels = []
    mfccs = []

    for f in fileList:
        if f.endswith('.wav'):

            wave, sr = librosa.load(RecordingsPath + f, mono=True, sr=None)
            label = f.split('_')[0]

            mfccs.append(wav2mfcc(wave, sr))
            labels.append(label)

            if augmentDataSet:
                noisyData = addNoise(wave)
                mfccs.append(wav2mfcc(noisyData, sr))
                labels.append(label)

                # mutedData = decreaseAmplitube(wave)
                # mfccs.append(wav2mfcc(mutedData, sr))
                # labels.append(label)

    mfccs = np.asarray(mfccs)
    mfccs = mfccs.reshape((mfccs.shape[0], mfccs.shape[1], mfccs.shape[2], 1))

    return mfccs, to_categorical(labels)

def getData(augmentDataSet = True):

    RecordingsPath = './free-spoken-digit-dataset/recordings/'

    fileList = os.listdir(RecordingsPath)

    traininingAndValidationFileList , testFileList = train_test_split(fileList, test_size=0.1)
    trainingFileList, validationFileList = train_test_split(traininingAndValidationFileList, test_size=0.2)

    X_test, y_test = getDataSet(RecordingsPath, testFileList)
    X_validation, y_validation = getDataSet(RecordingsPath, validationFileList)
    X_training, y_training = getTrainingSet(RecordingsPath, trainingFileList, augmentDataSet)

    return  X_training, y_training, X_validation, y_validation, X_test, y_test


# def get_all():
#     X_training, y_training, X_validation, y_validation, X_test, y_test = getData(False)
    
    
#     # dim_1 = X_training[0].shape[1]
#     # dim_2 = X_training[0].shape[2]
#     # channels = 1
#     # noOfClasses = 10

#     # # X = mfccs
#     # # X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
#     # # y = labels

#     # # input_shape = (dim_1, dim_2, channels)

#     # model = get_cnn_model(input_shape, noOfClasses)

#     return X_training, y_training, X_validation, y_validation, X_test, y_test


def showNoiseInfluence():
    wave, sr = librosa.load('./free-spoken-digit-dataset/recordings/0_george_0.wav', mono=True, sr=None)
    mfcc_original = wav2mfcc(wave, sr)
    noisyData = addNoise(wave)
    mfcc_noise = wav2mfcc(noisyData, sr)

    amplifiedData = increaseAmplitube(wave)
    mfcc_apl = wav2mfcc(amplifiedData, sr)
    attunedData = decreaseAmplitube(wave)
    mfcc_att = wav2mfcc(attunedData, sr)
    
    fig, ax = plt.subplots()

    fig2, ax2 = plt.subplots()
    
    fig3, ax3 = plt.subplots()
    
    fig4, ax4 = plt.subplots()
    img1 = librosa.display.specshow(mfcc_original, x_axis='time', ax=ax)
    img2 = librosa.display.specshow(mfcc_noise, x_axis='time', ax=ax2)
    img3 = librosa.display.specshow(mfcc_apl, x_axis='time', ax=ax3)
    img4 = librosa.display.specshow(mfcc_att, x_axis='time', ax=ax4)


    fig.colorbar(img1, ax=ax)
    fig2.colorbar(img2, ax=ax2)
    fig3.colorbar(img3, ax=ax3)
    fig4.colorbar(img4, ax=ax4)

    plt.show()

def foo():
    wave, sr = librosa.load('./free-spoken-digit-dataset/recordings/0_george_0.wav', mono=True, sr=None)

    # S = librosa.feature.melspectrogram(y=wave, sr=sr,hop_length=128)

    # fig, ax = plt.subplots()

    # S_dB = librosa.power_to_db(S, ref=np.max)

    # img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)

    # fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # ax.set(title='Mel-frequency spectrogram')

    # mfcc = librosa.feature.mfcc(S = S_dB, n_mfcc=40)

    # fig2, ax2 = plt.subplots()
    # img1 = librosa.display.specshow(mfcc, x_axis='time', ax=ax2)
    # fig2.colorbar(img1, ax=ax2)


    mfcc2 = librosa.feature.mfcc(wave, sr, hop_length=128, n_mfcc=40)

    fig3, ax3 = plt.subplots()
    img3 = librosa.display.specshow(mfcc2, x_axis='time', ax=ax3)
    fig3.colorbar(img3, ax=ax3)

    # mfcc_original = wav2mfcc(wave, sr)
    # noisyData = addNoise(wave)
    # mfcc_noise = wav2mfcc(noisyData, sr)

    # amplifiedData = increaseAmplitube(wave)
    # mfcc_apl = wav2mfcc(amplifiedData, sr)
    # attunedData = decreaseAmplitube(wave)
    # mfcc_att = wav2mfcc(attunedData, sr)
    
    # fig, ax = plt.subplots()

    # fig2, ax2 = plt.subplots()
    
    # fig3, ax3 = plt.subplots()
    
    # fig4, ax4 = plt.subplots()
    # img1 = librosa.display.specshow(mfcc_original, x_axis='time', ax=ax)
    # img2 = librosa.display.specshow(mfcc_noise, x_axis='time', ax=ax2)
    # img3 = librosa.display.specshow(mfcc_apl, x_axis='time', ax=ax3)
    # img4 = librosa.display.specshow(mfcc_att, x_axis='time', ax=ax4)


    # fig.colorbar(img1, ax=ax)
    # fig2.colorbar(img2, ax=ax2)
    # fig3.colorbar(img3, ax=ax3)Alm
    # fig4.colorbar(img4, ax=ax4)

    plt.show()
def drawInputWaveLenHistogram():
    RecordingsPath = './free-spoken-digit-dataset/recordings/'

    fileList = os.listdir(RecordingsPath)

    lengths = []
    for f in fileList:
        if f.endswith('.wav'):

            wave, sr = librosa.load(RecordingsPath + f, mono=True, sr=None)
            lengths.append(len(wave))
    
    _ = plt.hist(lengths)
    plt.show()


if __name__ == '__main__':
    # drawInputWaveLenHistogram()
    showNoiseInfluence()
