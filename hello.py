import cv2

cascPath = "/home/prateeksha/code/Pr4t3/DriverDrowsinessDetector/notebooks/haarcascade_frontalface_default.xml"
smilePath = "/home/prateeksha/code/Pr4t3/DriverDrowsinessDetector/notebooks/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
smileCascade = cv2.CascadeClassifier(smilePath)

arrival = '/home/prateeksha/code/Pr4t3/DriverDrowsinessDetector/raw_data/cropped_face/yawn_1.jpg'
finalpath= '/home/prateeksha/code/Pr4t3/DriverDrowsinessDetector/raw_data/cropped_mouth/yawn_1.jpg'

path ='/home/prateeksha/code/Pr4t3/DriverDrowsinessDetector/raw_data/image/yawn_1.jpg'
frame= cv2.imread(path)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

cv2.imwrite(arrival, roi_gray)         
smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        crop_img = roi_gray[sy:sy+sh, sx:sx+sw]
        cv2.imwrite(finalpath, crop_img)