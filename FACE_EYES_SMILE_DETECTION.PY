# Importowanie Bibliotek
import cv2
import numpy as np
import time

# Wyznaczenie Classifaierow  
face_cascade = cv2.CascadeClassifier("/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/haarcascades/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("/haarcascades/haarcascade_smile.xml")

# Wybranie zrudla obrazu
cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# Tworzenie niebieskiego kwadratu wokol glowy 
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        smile = smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.3,
        minNeighbors=50,
        minSize=(25, 25),
        )

        # Tworzenie Zielonych kwadratów wokol oczu
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        # Tworzenie zoltego prostokonta wokol usmiehcu i wykonanie polecen po usmiechu
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,255), 2)
            from PIL import Image
            im = Image.open("smile.png") # Ścieżka obrazu który ma się otworzyć po uśmiechnięciu
            im.show()
            
# Otworzenie Kamerki
    cv2.imshow('img', img)
# Zamykanie Okna Przyciskiem ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
