import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
lipCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
       # cv2.putText(frame,'Face',(x, y), font, 2,(255,0,0),5)
        cv2.putText(frame, str(x+w/2)+','+str(y+h/2), (x, y), font, 2, (255, 0, 0), 3)
        #print(frame.shape)
    lip = lipCascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (sx, sy, sw, sh) in lip:
        cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
        cv2.putText(frame,'Lips',(x + sx,y + sy), 1, 1, (0, 0, 255), 1)

    eyes = eyeCascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.16,
        minNeighbors=25,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
                                       )
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()