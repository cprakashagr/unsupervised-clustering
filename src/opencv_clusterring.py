import os
import shutil

import cv2
import imutils
import numpy as np

SZ = 20
bin_n = 16


def calculate_hog(img):
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
    Z = np.float32(np.arange(0).reshape(0, 64))
    baseFolder = '/Users/cprakashagr/Pictures/GreedyGame'
    allFiles = os.listdir(baseFolder)
    handledFiles = []
    for file in allFiles:
        if file.startswith('.') or os.path.isdir(file):        # .DS_STORE  && Dir
            continue
        filePath = os.path.join(baseFolder, file)
        try:
            img = cv2.imread(filePath)
            img = imutils.resize(img, height=min(200, img.shape[0]))
            hog = calculate_hog(img)
            Z = np.vstack((Z, hog))
            handledFiles.append(file)
        except:
            pass

    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    ret, label, center = cv2.kmeans(Z, 10, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    for i in range(len(handledFiles)):
        file = handledFiles[i]
        oldFile = os.path.join(baseFolder, file)

        currentLabel = label.ravel()[i]
        newPath = baseFolder+'/'+str(currentLabel)+'/'

        currentHog = Z[i]

        dist = np.linalg.norm(currentHog - center[currentLabel])

        shutil.copyfile(oldFile, newPath+str(dist))

if __name__ == '__main__':
    main()


'''
I -->       10
II          1000

'''
