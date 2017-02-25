import cv2
import numpy as np
import os

from os.path import isdir, isfile


'''
This is the blue print on SVM supervised.
  --  Under-development (& not tested)
'''


class SVM():
    def __init__(self):
        self.svm = cv2.ml.SVM_create()
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setC(2.67)
        self.svm.setGamma(5.383)

    def train(self, trainData, trainResponse):
        self.svm.train(trainData, cv2.ml.ROW_SAMPLE, trainResponse)

    def predict(self, testData):
        return self.svm.predict(testData)

    def save(self):
        self.svm.save('model.dat')

SZ = 20
bin_n = 16

trainX = []
trainY = []

testX = []
testY = []


def createSamples(baseFolder, isTrain):
    folders = [f for f in os.listdir(baseFolder) if isdir(os.path.join(baseFolder, f))]

    for folder in folders:
        newPath = os.path.join(baseFolder, folder)
        allFiles = [f for f in os.listdir(newPath) if isfile(f)]

        for file in allFiles:
            img = cv2.imread(file, 0)
            hog = calculateHog(img)
            if isTrain:
                trainX.append(np.float32(hog).reshape(-1, 64))
                trainY.append(int(file))
            else:
                testX.append(np.float32(hog).reshape(-1, 64))
                testY.append(file)


def calculateHog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


def main():

    svm = SVM()

    baseFolder = '/Users/cprakashagr/Pictures/GreedyGame'

    # Creating training Data
    createSamples(baseFolder, isTrain=True)

    # Training
    '''
    svm.train(trainData=trainX, trainResponse=trainY)
    svm.save()

    # Creating Test Data
    testBaseFolder = baseFolder + '/test'
    createSamples(testBaseFolder, isTrain=False)

    result = svm.predict(testData=testX)
    '''


if __name__ == '__main__':
    main()
