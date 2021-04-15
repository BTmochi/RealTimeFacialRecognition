from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            img = frame
            grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            front_face_list = face_cascade.detectMultiScale(grayscale_img, minSize=(100, 100))

            for (x, y, w, h) in front_face_list:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=10)

            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
