# Impor
from flask import Flask, render_template, request, send_from_directory


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
