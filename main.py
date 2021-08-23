import glob,random
import cv2
import numpy as np

totalImages = glob.glob("./data/*")
def processImg(img,lth,uth):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    low_threshold = lth
    high_threshold = uth
    img = cv2.Canny(blur_gray, low_threshold, high_threshold)
    contours, hierarchy = cv2.findContours(img,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
for im in totalImages:
    # white color mask
    final = cv2.imread(im) #origional image
    img = final*0
    puppyFrames = final*0

    contours = processImg(final,50,150)

    img = final*0
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)

    contours = processImg(img,0,255)
    img = final * 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if(hull_area == 0):
            continue
        solidity = float(area) / hull_area

        equi_diameter = np.sqrt(4 * area / np.pi)
        # print(cnt)
        if(len(cnt)>4):
            (x, y), (MA, ma), Orientation = cv2.fitEllipse(cnt)

            if(cv2.contourArea(cnt)>4000 and equi_diameter > 100 and extent > 0.7):
                print(
                    " Width = {}  Height = {} area = {}  aspect ration = {}  extent  = {} solidity = {} equi_diameter = {} orientation = {} ".format(
                        w, h, area, aspect_ratio, extent, solidity, equi_diameter, Orientation))
                contours_poly = cv2.approxPolyDP(cnt, 3, True)
                boundRect = cv2.boundingRect(contours_poly)
                hull = cv2.convexHull(cnt)
                cv2.drawContours(img, [hull], -1, (255, 255, 255), -1)
                puppy = cv2.imread(random.choice(glob.glob('./puppyImages/*')))
                puppy = cv2.resize(puppy, (boundRect[2],boundRect[3]))
                puppyFrames[int(boundRect[1]):int(boundRect[1] + boundRect[3]), int(boundRect[0]):int(boundRect[0] + boundRect[2])] = puppy
                cv2.polylines(img, pts=[cnt], color=(255, 255, 255), isClosed=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mult = cv2.multiply(img, puppyFrames)
    (B1, G1, R1) = cv2.split(puppyFrames)
    B1 = cv2.bitwise_and(B1, B1, mask=img)
    G1 = cv2.bitwise_and(G1, G1, mask=img)
    R1 = cv2.bitwise_and(R1, R1, mask=img)
    # res = cv2.merge([B1, G1, R1])

    img = cv2.bitwise_not(img)
    (B2, G2, R2) = cv2.split(final)
    B2 = cv2.bitwise_and(B2, B2, mask=img)
    G2 = cv2.bitwise_and(G2, G2, mask=img)
    R2 = cv2.bitwise_and(R2, R2, mask=img)
    B = cv2.add(B1,B2)
    G = cv2.add(G1, G2)
    R = cv2.add(R1, R2)

    res = cv2.merge([B, G, R])

    cv2.imwrite('./swappedImages/' + im.split('/')[-1], res)