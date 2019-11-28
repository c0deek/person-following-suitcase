import cv2
import numpy as np

body_cascade = cv2.CascadeClassifier('cascades/haarcascade_lowerbody.xml')

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://bit.ly/2ORncQD')

hp = 0
xp = 0
yp = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    body = body_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in body:

        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 2)

        roi = img[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([100,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange (hsv, lower_blue, upper_blue)
        bluecnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(bluecnts)>0:
            blue_area = max(bluecnts, key=cv2.contourArea)
            (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
            cv2.rectangle(roi,(xg,yg),(xg+wg, yg+hg),(255,0,0),2)

            if xg<xp:
                print("right: ", str(xp-xg)+" px/frame |", end='')
            elif xg>xp:
                print(" left: ", str(xg-xp)+" px/frame |", end='')
            else:
                print(' same: ', str(xp-xg)+" px/frame |", end='')
            xp = xg

            if yg>yp:
                print("down: ", str(yg-yp)+" px/frame |",end='')
            elif yg<yp:
                print(" top: ", str(yp-yg)+" px/frame |", end='')
            else:
                print('same: ', str(yg-yp)+" px/frame |", end='')
            yp = yg


            if hg<hp:
                print('far', str(hp-hg)+" px/frame |")
            elif hg>hp:
                print('near', str(hg-hp)+" px/frame |")
            else:
                print('same', str(hg-hp)+" px/frame |")
            hp = hg






    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
