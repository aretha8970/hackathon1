from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_trees(image_path):

    results = model(image_path)

    tree_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # YOLO class name
            label = model.names[cls]

            # We treat plants/trees as vegetation
            if label in ["potted plant", "plant"]:
                tree_count += 1

    return tree_count


@app.route("/", methods=["GET", "POST"])
def index():

    tree_count = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)
            image_path = path
            tree_count = detect_trees(path)


    return render_template("index.html", tree_count=tree_count, image=image_path)


if __name__ == "__main__":
    app.run(debug=True)