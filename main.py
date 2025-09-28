from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from typing import List, Dict, Any, Optional
import io
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Importar funciones de base de datos
from app.database import (
    save_face_to_db,      # <-- Usa este nombre
    delete_face_from_db,  # <-- Usa este nombre
    load_known_faces_from_db
    # get_face_embeddings  # Solo si la agregas en database.py
)

# Cargar variables de entorno
load_dotenv()

app = FastAPI(
    title="API de Reconocimiento Facial",
    description="API para reconocimiento facial y gestión de personas",
    version="1.0.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite cualquier origen (ajustar para producción)
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todos los encabezados
)

# Almacenamiento en memoria para rostros conocidos (caché de la base de datos)
known_faces: List[Dict[str, Any]] = []

def load_known_faces():
    """Carga los rostros conocidos desde la base de datos a la memoria."""
    global known_faces
    known_faces = [] # Limpiar caché actual
    try:
        db_faces = load_known_faces_from_db()
        for face_id, name, embedding in db_faces:
            # Asegurarse de que el embedding es una lista o tupla antes de convertir a np.array
            if isinstance(embedding, (list, tuple)):
                 known_faces.append({
                    'id': face_id,
                    'name': name,
                    'encoding': np.array(embedding)
                 })
            else:
                 # Manejar caso donde el embedding no es una lista (posiblemente NULL o formato incorrecto)
                 print(f"Advertencia: Embedding para '{name}' (ID: {face_id}) no es una lista. Saltando.")


        print(f"Cargados {len(known_faces)} rostros conocidos desde la base de datos")
    except Exception as e:
        print(f"Error cargando rostros conocidos: {e}")
        # En un entorno de producción, esto debería ser un logger.error


@app.on_event("startup")
async def startup_event():
    """Carga los rostros conocidos cuando la aplicación arranca."""
    print("Iniciando API de Reconocimiento Facial...") # Usar logger en producción
    load_known_faces()
    print("Carga inicial de rostros conocidos completada.") # Usar logger en producción

@app.post("/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Registra un nuevo rostro en el sistema.

    - **name**: Nombre de la persona
    - **file**: Archivo de imagen que contiene el rostro
    """
    try:
        print(f"Iniciando registro facial para: {name}") # Usar logger
        # print(f"Archivo recibido: {file.filename}, size: {file.size} bytes") # Usar logger

        # Leer y procesar la imagen
        contents = await file.read()
        # print(f"Leídos {len(contents)} bytes del archivo subido") # Usar logger

        try:
            #face_recognition.load_image_file maneja bytes directamente
            image = face_recognition.load_image_file(io.BytesIO(contents))
            print("Imagen cargada exitosamente") # Usar logger
        except Exception as e:
            print(f"Error cargando imagen: {str(e)}") # Usar logger.error
            raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

        # Encontrar todos los encodings de rostros en la imagen
        print("Detectando rostros en la imagen...") # Usar logger
        try:
            face_encodings = face_recognition.face_encodings(image)
            print(f"Encontrados {len(face_encodings)} rostro(s) en la imagen") # Usar logger
        except Exception as e:
            print(f"Error detectando rostros: {str(e)}") # Usar logger.error
            raise HTTPException(status_code=400, detail=f"Error detectando rostros: {str(e)}")

        if not face_encodings:
            print("No se encontraron rostros en la imagen") # Usar logger.warning
            raise HTTPException(status_code=400, detail="No se encontraron rostros en la imagen")

        # Usar el primer rostro encontrado
        face_encoding = face_encodings[0]
        print("Encoding facial generado exitosamente") # Usar logger

        # Guardar en la base de datos
        print("Guardando rostro en la base de datos...") # Usar logger
        try:
            # save_face_embedding espera el embedding como lista
            person_id = save_face_to_db(name, face_encoding.tolist())
            print(f"Rostro guardado exitosamente con ID: {person_id}") # Usar logger
        except Exception as e:
            print(f"Error de base de datos: {str(e)}") # Usar logger.error
            raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")

        # Actualizar caché en memoria
        print("Actualizando caché facial en memoria...") # Usar logger
        try:
            load_known_faces()
            print(f"Caché en memoria actualizada exitosamente. Total rostros conocidos: {len(known_faces)}") # Usar logger
        except Exception as e:
            print(f"Advertencia: No se pudo actualizar la caché en memoria: {str(e)}") # Usar logger.warning

        return {
            "status": "success",
            "person_id": person_id,
            "name": name,
            "message": "Rostro registrado exitosamente"
        }

    except HTTPException as he:
        # Relanzar excepciones HTTP tal cual
        print(f"Excepción HTTP: {he.detail}") # Usar logger.error
        raise he
    except Exception as e:
        # Registrar el error completo para depuración
        import traceback
        error_details = traceback.format_exc()
        print(f"Error inesperado en register_face: {error_details}") # Usar logger.critical
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Reconoce rostros en la imagen subida.

    - **file**: Archivo de imagen que contiene los rostros a reconocer
    """
    try:
        # Leer y procesar la imagen
        contents = await file.read()
        image = face_recognition.load_image_file(io.BytesIO(contents))

        # Encontrar todos los encodings de rostros en la imagen
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            raise HTTPException(status_code=400, detail="No se encontraron rostros en la imagen")

        # Comparar con rostros conocidos
        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            if not known_faces:
                # Si no hay rostros conocidos cargados, no hay coincidencias posibles
                name = "Desconocido"
                person_id = None
                confidence = 0.0
                best_match_distance = None # No hay distancia si no hay rostros conocidos
            else:
                # Obtener distancias a todos los rostros conocidos
                known_encodings_list = [face['encoding'] for face in known_faces]
                face_distances = face_recognition.face_distance(
                    known_encodings_list,
                    face_encoding
                )
                # Encontrar el mejor match
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]

                # Usar un umbral para determinar una coincidencia (menor distancia es más similar)
                face_match_threshold = 0.6 # Umbral común, puede ajustarse
                matches = [best_match_distance <= face_match_threshold]

                if matches[0]:  # Si tenemos una coincidencia
                    name = known_faces[best_match_index]['name']
                    person_id = known_faces[best_match_index]['id']
                    confidence = 1.0 - best_match_distance  # Convertir a score de confianza (0-1)
                else:
                    name = "Desconocido"
                    person_id = None
                    confidence = 0.0

            top, right, bottom, left = face_location
            results.append({
                "person_id": person_id,
                "name": name,
                "confidence": float(confidence),
                "distance_to_best_match": float(best_match_distance) if best_match_distance is not None else None, # Añadir distancia
                "location": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }
            })

        return {
            "status": "success",
            "results": results,
            "total_faces": len(results)
        }

    except Exception as e:
        # Registrar el error completo para depuración
        import traceback
        error_details = traceback.format_exc()
        print(f"Error inesperado en recognize_face: {error_details}") # Usar logger.critical
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.delete("/faces/{persona_id}")
async def delete_persona(persona_id: int):
    """
    Elimina una persona registrada del sistema (borrado lógico si la función lo soporta).

    - **persona_id**: ID de la persona a eliminar
    """
    try:
        # delete_face debe retornar True si fue exitoso, False si no se encontró
        success = delete_face_from_db(persona_id)
        if success:
            # Actualizar caché en memoria
            load_known_faces()
            return {
                "status": "success",
                "message": f"Persona con ID {persona_id} eliminada exitosamente."
            }
        else:
            raise HTTPException(status_code=404, detail=f"Persona con ID {persona_id} no encontrada.")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error inesperado al eliminar persona con ID {persona_id}: {error_details}") # Usar logger.critical
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al eliminar persona: {str(e)}")

@app.get("/faces")
async def list_faces():
    """
    Lista todos los rostros registrados en el sistema (cargados en caché).
    """
    try:
        # Devolver solo ID y nombre de la caché en memoria
        return {
            "status": "success",
            "count": len(known_faces),
            "faces": [
                {
                    "id": face['id'],
                    "name": face['name']
                }
                for face in known_faces
            ]
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error inesperado al listar rostros conocidos: {error_details}") # Usar logger.critical
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al listar rostros: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud para comprobar si la API está corriendo.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "known_faces_count": len(known_faces),
        "message": "API de Reconocimiento Facial está operativa."
    }

if __name__ == "__main__":
    import uvicorn
    # Ejecutar la API usando uvicorn
    print("Iniciando servidor Uvicorn para la API de Reconocimiento Facial.") # Usar logger
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)