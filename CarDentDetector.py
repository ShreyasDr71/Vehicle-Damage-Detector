from flask import Flask, render_template, request, redirect, url_for, Response
import os
from werkzeug.utils import secure_filename
import cv2
import math
import cvzone
from ultralytics import YOLO

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Directory for uploads
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLOv8 model
MODEL_PATH = "Weights/best.pt"  # Replace with your YOLO model's path
model = YOLO(MODEL_PATH)

# Define class names
class_labels = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process image
def process_image(file_path):
    img = cv2.imread(file_path)
    results = model(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            if conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{os.path.basename(file_path)}')
    cv2.imwrite(output_path, img)
    return output_path

# Process video
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{os.path.basename(file_path)}')

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                if conf > 0.3:
                    cvzone.cornerRect(frame, (x1, y1, w, h), t=2)
                    cvzone.putTextRect(frame, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))
        
        out.write(frame)

    cap.release()
    out.release()
    return output_path

# Live feed generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil(box.conf[0] * 100) / 100
                    cls = int(box.cls[0])

                    if conf > 0.3:
                        cvzone.cornerRect(frame, (x1, y1, w, h), t=2)
                        cvzone.putTextRect(frame, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process file based on its type
        if filename.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
            output_path = process_image(file_path)
        elif filename.split('.')[-1].lower() == 'mp4':
            output_path = process_video(file_path)
        else:
            return "Unsupported file type."

        return render_template(
            'result.html',
            output_file=url_for('static', filename=f'uploads/{os.path.basename(output_path)}'),
            file_type=filename.split('.')[-1].lower()
        )

    return redirect(request.url)

@app.route('/live_feed')
def live_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
