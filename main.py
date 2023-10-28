import cv2
import numpy as np
# import imutils
import face_recognition
import os
from datetime import datetime
from flask import Flask, render_template, Response, request


# Initialize the objects
app = Flask(__name__)
@app.route('/')
def index():
    # form = form
    # #Video streaming home page
    # if request.method == 'POST':
    #     if request.form.get('action1') == 'VALUE1':
    #         pass # do something
    #     elif  request.form.get('action2') == 'VALUE2':
    #         pass # do something else
    #     else:
    #         pass # unknown
    # elif request.method == 'GET':
    #     return render_template('index.html', form=form)
    
    return render_template("index.html")
    # return render_template('index.html')


#### from securaa
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Atten.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')
#### end Securaa

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
# Load the pretrained model
face_cascade.load(cv2.samples.findFile("static/haarcascade_frontalface_alt.xml"))



def gen(video):
    while True:
        # success, image = video.read()
        # frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)

        # faces = face_cascade.detectMultiScale(frame_gray)

        # for (x, y, w, h) in faces:
        #     center = (x + w//2, y + h//2)
        #     cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        #     image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #     faceROI = frame_gray[y:y+h, x:x+w]
        # ret, jpeg = cv2.imencode('.jpg', image)

        # frame = jpeg.tobytes()
        
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        success, image = video.read()
        # image = captureScreen()
        imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

            else:
                name = "Anonymous"
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
                cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
            # faceROI = imgS[y2, x1]
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
            
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
        
        
        
        

# @app.route('/video_feed')
# def video_feed():
# 		# Set to global because we refer the video variable on global scope, 
# 		# Or in other words outside the function
#     global video

# 		# Return the result on the web
#     return Response(gen(video),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed')
def video_feed():
    #Video streaming route
    return Response(gen(video),mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
    

if __name__ == '__main__':
     app.run(host='192.168.118.54',port='5000', debug=False,threaded = True)