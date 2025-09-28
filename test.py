from flask import Flask, jsonify
import requests
import os

app = Flask(__name__)

# URL base de la API
API_BASE_URL = "http://localhost:8000"

# --- Datos de prueba ---
REGISTER_IMAGE_PATH = "path/to/your/register_photo.jpg"  # ¡CAMBIA ESTA RUTA!
REGISTER_PERSON_NAME = "NombrePrueba"
RECOGNIZE_IMAGE_PATH = "path/to/your/recognize_photo.jpg"  # ¡CAMBIA ESTA RUTA!

# --- Funciones de prueba para cada endpoint ---

def test_status():
    """Prueba el endpoint /status/."""
    url = f"{API_BASE_URL}/status/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error al probar /status/: {e}"}

def test_register_face(name: str, image_path: str):
    """Prueba el endpoint /register_face/ subiendo un archivo."""
    url = f"{API_BASE_URL}/register_face/"
    if not os.path.exists(image_path):
        return {"error": f"La imagen para registrar no existe en la ruta: {image_path}"}

    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {'nombre': name}
            response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error al probar /register_face/: {e}"}

def test_recognize_face(image_path: str):
    """Prueba el endpoint /recognize_face/ subiendo un archivo."""
    url = f"{API_BASE_URL}/recognize_face/"
    if not os.path.exists(image_path):
        return {"error": f"La imagen para reconocer no existe en la ruta: {image_path}"}

    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error al probar /recognize_face/: {e}"}

def test_list_known_faces():
    """Prueba el endpoint /list_known_faces/."""
    url = f"{API_BASE_URL}/list_known_faces/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error al probar /list_known_faces/: {e}"}

def test_delete_face(name: str):
    """Prueba el endpoint /delete_face/{nombre}."""
    url = f"{API_BASE_URL}/delete_face/{name}"
    try:
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error al probar /delete_face/: {e}"}

# --- Ruta principal que devuelve el HTML con los resultados ---
@app.route('/')
def index():
    """Página principal que muestra el estado y resultados de las pruebas."""
    status = test_status()
    register_result = test_register_face(REGISTER_PERSON_NAME, REGISTER_IMAGE_PATH)
    recognize_result = test_recognize_face(RECOGNIZE_IMAGE_PATH)
    list_faces_result = test_list_known_faces()

    # HTML dentro del archivo Python
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pruebas API Reconocimiento Facial</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ color: #2e3d49; }}
            pre {{ background-color: #f4f4f4; padding: 10px; }}
            .result {{ margin-top: 20px; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Pruebas de la API de Reconocimiento Facial</h1>

        <div class="result">
            <h2>Estado de la API:</h2>
            <pre>{status}</pre>
        </div>

        <div class="result">
            <h2>Registro de Rostro ({REGISTER_PERSON_NAME}):</h2>
            <pre>{register_result}</pre>
        </div>

        <div class="result">
            <h2>Resultado del Reconocimiento de Rostro:</h2>
            <pre>{recognize_result}</pre>
        </div>

        <div class="result">
            <h2>Lista de Rostros Conocidos:</h2>
            <pre>{list_faces_result}</pre>
        </div>

    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
