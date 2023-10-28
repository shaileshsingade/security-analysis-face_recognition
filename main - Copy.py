from flask import Flask, Response, render_template
import cv2

# Initialize the objects
app = Flask(__name__)
@app.route('/')
def index():
    #Video streaming home page
    return render_template('index.html')

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
# Load the pretrained model
face_cascade.load(cv2.samples.findFile("static/haarcascade_frontalface_alt.xml"))



def gen(video):
    while True:
        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y+h, x:x+w]
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