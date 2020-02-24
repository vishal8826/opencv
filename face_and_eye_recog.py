import cv2

fd = cv2.CascadeClassifier("xmls\haarcascade_frontalface_default.xml")
left_eye = cv2.CascadeClassifier("xmls\haarcascade_lefteye_2splits.xml")
right_eye = cv2.CascadeClassifier("xmls\haarcascade_righteye_2splits.xml")

cam = cv2.VideoCapture(0)
while True:
    bindex,img = cam.read()
    gimg = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)
    face = fd.detectMultiScale(gimg, 1.3,2)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),3)
        roi_gray = gimg[x:x+w,y:y+h]
        roi_img = img[x:x+w,y:y+h]
        l_eye = left_eye.detectMultiScale(roi_gray)
        r_eye = right_eye.detectMultiScale(roi_gray)
        for (lx,ly,lw,lh), (rx, ry, rw, rh) in zip(l_eye, r_eye):
            cv2.rectangle(roi_img,(lx,ly),(lx+lw,ly+lh),(0,255,0),3)
            cv2.rectangle(roi_img,(rx,ry),(rx+rw,ry+rh),(0,255,0),3)
        cv2.imshow('',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()