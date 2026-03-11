# Impor
from flask import Flask, render_template, request, send_from_directory
from imageai.Detection import ObjectDetection
# Menginstansiasi kelas deteksi objek
detector = ObjectDetection()
# Mengatur jalur ke model YOLOv3
model_path = "/content/yolov3.pt"
# Menginstal model YOLOv3 dan mengatur jalur ke file bobot
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
#  Memuat model
detector.loadModel()
detector.CustomObjects()

# Mengenali gambar input pada gambar yang dipilih dengan probabilitas minimal 30% dan menyimpannya ke file output baru
detections = detector.detectObjectsFromImage(
    input_image="/tree.jpg",
    output_image_path="test2.jpg",
    minimum_percentage_probability=30)

# Menampilkan gambar
from google.colab.patches import cv2_imshow
import cv2
img = cv2.imread('test2.jpg', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

def detect_trees(input_image, output_image, model_path):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(
        input_image=input_image,
        output_image_path=output_image,
        minimum_percentage_probability=30
    )
    return detections
def analyze_objects(detections):
    trees = []
    if len(detections) > 0:
      for detection in detections:
          if detection["name"] in ["tree"]:
              trees append(detection)
    return trees


# Menampilkan gambar
from google.colab.patches import cv2_imshow
import cv2

input_image = "/tree.jpg"
output_image = "test2.jpg"
detections = detect_trees(input_image, output_image, "/content/yolov3.pt")
trees = analyze_objects(detections)
amount_of_trees = len(trees)
if len(trees) > 0:
  print("Detected foods:")
  for obj in trees:
      print(obj["name"], " : ", obj["percentage_probability"], " : ", obj["box_points"])
      print(f"the total amount of trees is {amount_of_trees}")

img = cv2.imread('test2.jpg', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

app = Flask(__name__)

# Hasil formulir
@app.route('/', methods=['GET','POST'])
def index():

    if request.method == 'POST':
        # mendapatkan gambar yang dipilih
        selected_image = request.form.get('image-selector')


        return render_template('index.html', 
                               # Menampilkan gambar yang dipilih
                               selected_image=selected_image, 

                               )
    else:
        # Menampilkan gambar pertama secara default
        return render_template('index.html', selected_image='logo.svg')


@app.route('/static/img/<path:path>')
def serve_images(path):
    return send_from_directory('static/img', path)

app.run(debug=True)
