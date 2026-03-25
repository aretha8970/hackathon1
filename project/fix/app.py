"""
Tree Detection Web App - Flask Backend
User upload gambar → deteksi pohon → tampilkan hasil + jumlah pohon + gambar anotasi
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import io
import base64
from datetime import datetime


# Konfigurasi
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "best.pt")  # Ganti dengan path model kamu
UPLOAD_DIR   = "uploads"
RESULT_DIR   = "results"
MIN_CONF     = 0.25
MAX_IMG_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXT  = {"jpg", "jpeg", "png", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app   = Flask(__name__, static_folder="static")
CORS(app)  # Izinkan request dari frontend

# Load model sekali saat startup
print(f"🌲 Loading model dari: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"✅ Model siap. Label: {model.names}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_detection(image_path: str) -> dict:
    """Jalankan deteksi YOLOv8 dan kembalikan hasil lengkap."""
    results = model.predict(
        source=image_path,
        conf=MIN_CONF,
        verbose=False
    )
    result = results[0]

    # Gambar anotasi
    annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR → RGB
    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path     = os.path.join(RESULT_DIR, result_filename)
    annotated.save(result_path, quality=90)

    # Kumpulkan data deteksi per objek
    detections = []
    label_count: dict[str, int] = {}

    for box in result.boxes:
        cls_id     = int(box.cls[0])
        label      = model.names[cls_id]
        confidence = round(float(box.conf[0]) * 100, 1)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "label":      label,
            "confidence": confidence,
            "box":        [x1, y1, x2, y2]
        })

        label_count[label] = label_count.get(label, 0) + 1

    # Hitung total pohon (semua label dianggap pohon, atau filter spesifik)
    total_trees = len(detections)

    return {
        "total_trees":    total_trees,
        "label_count":    label_count,   # {"tree": 5, "palm": 2, ...}
        "detections":     detections,
        "annotated_image": image_to_base64(result_path),
        "result_filename": result_filename,
    }



# Routes

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    POST /api/detect
    Body: multipart/form-data dengan field 'image'
    Response: JSON dengan hasil deteksi
    """
    # Validasi input
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Format file tidak didukung. Gunakan: {', '.join(ALLOWED_EXT)}"}), 400

    if request.content_length and request.content_length > MAX_IMG_SIZE:
        return jsonify({"error": "Ukuran file terlalu besar (maks 10 MB)."}), 413

    # Simpan file sementara
    tmp_filename = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path     = os.path.join(UPLOAD_DIR, tmp_filename)
    file.save(tmp_path)

    try:
        result = run_detection(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Deteksi gagal: {str(e)}"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify({
        "success":         True,
        "total_trees":     result["total_trees"],
        "label_count":     result["label_count"],
        "detections":      result["detections"],
        "annotated_image": result["annotated_image"],  # Base64 JPEG
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "model":      MODEL_PATH,
        "labels":     model.names,
        "min_conf":   MIN_CONF,
        "timestamp":  datetime.utcnow().isoformat()
    })



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
