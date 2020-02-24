import cv2

fd = cv2.CascadeClassifier("xmls\haarcascade_frontalface_default.xml")

img = cv2.imread('image\me.jpg')
gimg = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)
face = fd.detectMultiScale(gimg, 1.3,2) #width height in face
for (x,y,w,h) in face:
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow('',img)
    if cv2.waitKey(0) & 0xFF==ord('q'):
        break
            
cv2.destroyAllWindows()

