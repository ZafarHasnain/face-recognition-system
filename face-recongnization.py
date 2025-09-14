# import cv2
# face_cap = cv2.CascadeClassifier("C:/Users/Zafar Hasnain/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default")
# col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)

# faces = face_cap.detectMultiScale(
#     col,
#     scaleFactor= 1.1,
#     minNeighbors=5,
#     minSize=(30,30),
#     flags=cv2,
# )
# for(x,y,h,w) in faces:
#     cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
# videos_cap = cv2.VideoCapture(0)
# while True:
#     ret, video_data = videos_cap.read()
#     cv2.imshow("video_live",video_data)
#     if cv2.waitKey(10) == ord("c"):
#         break

# videos_cap.release()    


import cv2

# Load the Haar cascade automatically
face_cap = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
videos_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = videos_cap.read()
    if not ret:
        print("Cannot access camera")
        break

    # Convert each frame to grayscale AFTER you have the frame
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("video_live", video_data)

    # Press 'c' to close
    if cv2.waitKey(10) & 0xFF == ord("c"):
        break

videos_cap.release()
cv2.destroyAllWindows()
