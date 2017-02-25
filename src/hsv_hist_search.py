import argparse
import os
import sys

import cv2
import numpy as np

bins = (8, 12, 3)
handledFiles = []
allFeatures = []
baseFolder = '/Users/cprakashagr/Pictures/GreedyGame'


def search(queryFeature):
    results = {}

    for i in range(len(allFeatures)):
        featureToTest = allFeatures[i]
        # "Correlation", 0;  "Chi-Squared", 1;  "Intersection", 2;  "Hellinger", 3
        d = cv2.compareHist(np.asarray(queryFeature), np.asarray(featureToTest), 3)
        if d > 0.55:
            fullPath = os.path.join(baseFolder, handledFiles[i])
            key = str(d) + handledFiles[i]
            results[key] = fullPath

    return results


def describe(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []

    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

    hX = int(cX / 2); hY = int(cY / 2)
    sX = cX - hX; sY = cY - hY; eX = cX + hX; eY = cY + hY
    centRect = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(centRect, (sX, sY), (eX, eY), 255, -1)

    for (startX, endX, startY, endY) in segments:
        cornerMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, centRect)

        hist = histogram(image, cornerMask)
        features.extend(hist)

    hist = histogram(image, centRect)
    features.extend(hist)
    return features


def histogram(image, mask):
    hist = cv2.calcHist([image], [0, 1, 2], mask, bins,
                        [0, 180, 0, 256, 0, 256])
    return hist


def preprocessing():
    allFiles = os.listdir(baseFolder)
    for file in allFiles:
        if file.startswith('.') or os.path.isdir(file):        # .DS_STORE  && Dir
            continue
        filePath = os.path.join(baseFolder, file)
        try:
            img = cv2.imread(filePath)
            feature = describe(img)
            allFeatures.append(feature)
            handledFiles.append(file)
        except:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", required=True, help="Path to the query image")
    args = vars(ap.parse_args())

    preprocessing()

    query = cv2.imread(args["query"])
    try:
        features = describe(query)
        results = search(features)

        cv2.imshow("Query", query)

        print('Total Matching Found: ' + str(len(results)))
        for key, value in results.items():
            img = cv2.imread(value)
            cv2.imshow(key, img)
            cv2.waitKey(0)

    except ValueError as e:
        print(e)
        print(sys.exc_info()[0])

if __name__ == '__main__':
    main()
