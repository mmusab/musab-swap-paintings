import glob,random
import cv2
import numpy as np

totalImages = glob.glob("./data/*")
def processImg(img,lth,uth):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scaling the image for binary manipulation
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) # bluring 
    low_threshold = lth
    high_threshold = uth
    img = cv2.Canny(blur_gray, low_threshold, high_threshold) # canny edge detection to figure out painting borders
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # finding contours from edge image
    return contours # returning contours for postprocessing

for im in totalImages:
    orig = cv2.imread(im) #origional image
    img = orig*0 # black background for making mask for puppy frames to fit inside painting borders
    puppyFrames = orig*0 # black background for placing puppy images

    contours = processImg(orig,50,150) # get contours from grayscale painting image

    cv2.drawContours(img, contours, -1, (255, 255, 255), 3) # draw contours on a black background with white line

    contours = processImg(img,0,255) # get contours from binary contour image to get more refine border information
    img = orig * 0 # reinitialize to black background to construct it into mask
    for cnt in contours: # going through all the contours for specific image to calculate their geometric information
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h # aspect ration showing relation between length and width of contour bounding

        area = cv2.contourArea(cnt) # total area of contour polygon
        rect_area = w * h # total area of bounding recangle of contour
        extent = float(area) / rect_area # showing how much its close to rectangular shape

        hull = cv2.convexHull(cnt) # finding hull to match it to reactngle
        hull_area = cv2.contourArea(hull) # calculating hull area
        if(hull_area == 0): # skip to next contour if area is zero
            continue
        equi_diameter = np.sqrt(4 * area / np.pi) # estimating the form of contour

        if(cv2.contourArea(cnt)>4000 and equi_diameter > 100 and extent > 0.7): # checking area,equidiameter and extent to estimate if contour is in fact painting
            print(" Width = {}  Height = {} area = {}  aspect ration = {}  extent  = {} orientation = {} ".format(
                    w, h, area, aspect_ratio, extent, equi_diameter))
            contours_poly = cv2.approxPolyDP(cnt, 3, True) # getting contour polygon
            boundRect = cv2.boundingRect(contours_poly) # getting contour bounding recangle representing painting boundary
            hull = cv2.convexHull(cnt) # calculating hull to estimate positioning and oriantation of painting with tilt
            cv2.drawContours(img, [hull], -1, (255, 255, 255), -1) # drawing that hull(paiting) on black background image making mask for specific painting area
            puppy = cv2.imread(random.choice(glob.glob('./puppyImages/*'))) # getting random puppy image from directory
            puppy = cv2.resize(puppy, (boundRect[2],boundRect[3])) # resizing it to size of paiting rectanle
            puppyFrames[int(boundRect[1]):int(boundRect[1] + boundRect[3]), int(boundRect[0]):int(boundRect[0] + boundRect[2])] = puppy # plaing puppy image on specific location on black image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # getting single channel of mask
    # splitting puppy frame placement image to apply mask on all three channels
    (B1, G1, R1) = cv2.split(puppyFrames)
    B1 = cv2.bitwise_and(B1, B1, mask=img)
    G1 = cv2.bitwise_and(G1, G1, mask=img)
    R1 = cv2.bitwise_and(R1, R1, mask=img)

    img = cv2.bitwise_not(img) # inverting mask to get background of painting except painting part

    # splitting origional image to apply inverted mask to get background for all three channels
    (B2, G2, R2) = cv2.split(orig)
    B2 = cv2.bitwise_and(B2, B2, mask=img)
    G2 = cv2.bitwise_and(G2, G2, mask=img)
    R2 = cv2.bitwise_and(R2, R2, mask=img)

    # adding all three channels of forground and background
    B = cv2.add(B1,B2)
    G = cv2.add(G1, G2)
    R = cv2.add(R1, R2)

    # merging all channels to get an image
    res = cv2.merge([B, G, R])

    #writing the swapped image in dir "swappedImages" with same image name
    cv2.imwrite('./swappedImages/' + im.split('/')[-1], res)