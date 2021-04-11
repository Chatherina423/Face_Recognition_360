import face_recognition
import pickle
import cv2
import os
import numpy as np
from speech import speak


# find path of xml file containing haarcascade file
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())

text1 ="Streaming started"
speak(text1)

video_capture = cv2.VideoCapture(0)

width  = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
print(width, height)


while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    # convert the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    face_locations =[]
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes

    face_found = False

    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values and True for the embeddings it matches closely
        # and False for rest


        distances = distances = face_recognition.face_distance(data["encodings"], encoding)
        min_value = min(distances)

        # set name =unknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match

        if min_value < 0.5:
            index = np.argmin(distances)
            name = data["names"][index]

        # update the list of names
        names.append(name)
        # loop over the recognized faces

        count_l = 0
        count_r = 0

        print(names)

        # list_of_names = '\n'.join(names)
        #
        # cv2.putText(frame, list_of_names, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for i in range(len(names)):
            cv2.putText(frame, names[i], (100, 100*i + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # for name in names:
        #     speak(name)

        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image

            if w>0:
                face_found = True
                center = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(frame, ".", center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                print("This is the center of face of" + " "  + name + ": " + str(center[0]) + ", " + str(center[1]))

                print(center[0])

                if center[0] < 640:
                    print("Location is Left")

                else:
                    print("Location is Right")


    if face_found == False:
        cv2.putText(frame, "Face Not Detected", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("finish")