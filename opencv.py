# Import opencv for camera processing
import cv2

# Load haarscade file to detect face
algo = "haarcascade_frontalface_default.xml" # algo file
haar_cascade = cv2.CascadeClassifier(algo) # read cascade file

# Initialize camera
cam = cv2.VideoCapture(0) 

# Start Feeding camera
while True:
    # read camera
    _,img = cam.read()
    
    # convert image to gray image
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # detectMultiScale i.e coordinates of face
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(46, 204, 113),2)
        cv2.putText(img,"Face Detected",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
    
    # Show image in window
    cv2.imshow("Face Mask Detection",img)
    
    # interrupt by q key
    key = cv2.waitKey(2)  & 0xFF
    if(key == ord('q') & 0x71):
        break

# release camera
cam.release()
cv2.destroyAllWindows()