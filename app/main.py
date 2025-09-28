from fastapi import FastAPI, HTTPException, Form, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import os
import uvicorn
import numpy as np
from typing import List, Optional
from pathlib import Path
import asyncio
import logging

# Importar nuestros módulos
from .database import (
    get_db_connection,
    load_known_faces_from_db,
    delete_face_from_db,
    save_face_to_db,
    get_db_status
)
from .photo_cleaner import cleanup_orphaned_photos
from .face_utils import (
    process_image,
    extract_face_embedding,
    recognize_face as recognize_face_util,
    is_face_duplicate,
    FaceRecognitionResult
)

# --- Configuración de la aplicación ---
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "reference_photos"

# Crear el directorio de imágenes si no existe
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Configurar registro
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Modelos Pydantic ---
class RecognitionResult(BaseModel):
    name: str
    is_known: bool
    distance: Optional[float] = None

# --- Inicializar FastAPI ---
app = FastAPI(
    title="API de Reconocimiento Facial",
    description="API para registrar y reconocer rostros usando PostgreSQL",
    version="1.0.0"
)

# Almacenamiento en memoria de los rostros conocidos
known_face_encodings: List[np.ndarray] = []
known_face_names: List[str] = []

# --- Eventos de la aplicación ---
@app.on_event("startup")
async def startup_event():
    """Se ejecuta cuando la aplicación FastAPI arranca."""
    print("\n" + "="*50)
    print("Iniciando API de Reconocimiento Facial...")
    print("="*50)

    # Verificar y mostrar la ruta de las fotos de referencia
    print(f"\nBuscando fotos de referencia en: {IMAGE_FOLDER.absolute()}")
    if not IMAGE_FOLDER.exists():
        print(f"¡Error! No se encontró el directorio: {IMAGE_FOLDER}")
    else:
        image_files = [f for f in IMAGE_FOLDER.glob('*')
                       if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"Se encontraron {len(image_files)} imágenes de referencia.")

    # Cargar los rostros conocidos desde la base de datos
    try:
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces_from_db()
        logger.info(f"[{len(known_face_names)}] rostros cargados desde la base de datos.")
        if known_face_names:
            logger.info(f"Personas reconocidas: {', '.join(known_face_names)}")

        print("\n" + "="*50)
    except Exception as e:
        logger.error(f"Error al cargar los rostros desde la base de datos: {e}")
        # Consider logging the traceback in production
        raise

    # Registrar automáticamente las imágenes en la carpeta reference_photos
    print("\nProcesando imágenes de referencia...")
    registered_count = 0
    error_count = 0

    for img_path in IMAGE_FOLDER.glob('*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        nombre = img_path.stem
        print(f"\nProcesando: {img_path.name}")

        # Leer la imagen
        try:
            with open(img_path, 'rb') as f:
                image_data = f.read()
        except Exception as e:
            print(f"  - ✗ Error al leer la imagen {img_path.name}: {str(e)}")
            error_count += 1
            continue

        # Verificar si la imagen ya está registrada
        if nombre in known_face_names:
            print(f"  - {nombre} ya está registrado. Actualizando...")
            # Consider adding update logic here if needed, currently it just skips registration
            # For now, it will be skipped and handled by the registration endpoint if called manually

        # Procesar y registrar el rostro (solo si no está ya en la caché)
        if nombre not in known_face_names:
            print(f"  - Procesando imagen: {img_path.name}")
            try:
                # Procesar la imagen primero
                rgb_image, process_info = process_image(image_data)

                if not process_info.get('success'):
                     error_msg = process_info.get('error', 'Error desconocido al procesar la imagen')
                     print(f"  - ✗ Error al procesar {img_path.name}: {error_msg}")
                     error_count += 1
                     continue

                # Extraer el embedding
                success, face_encoding, face_metadata = extract_face_embedding(rgb_image)

                if not success or face_encoding is None:
                    error_msg = face_metadata.get('error', 'No se pudo extraer el rostro')
                    print(f"  - ✗ Error al procesar {img_path.name}: {error_msg}")
                    error_count += 1
                    continue

                # Verificar si ya existe un rostro similar (usando la caché actual)
                is_duplicate, duplicate_name, _ = is_face_duplicate(
                    face_encoding,
                    known_face_encodings,
                    known_face_names
                )

                if is_duplicate:
                    print(f"  - ⚠ Rostro de '{nombre}' ya existe como '{duplicate_name}'. Omitiendo...")
                    error_count += 1
                    continue

                # Guardar en la base de datos
                person_id = save_face_to_db(nombre, face_encoding.tolist())
                print(f"  - ✓ '{nombre}' registrado exitosamente con ID: {person_id}")
                registered_count += 1

            except Exception as e:
                logger.error(f"  - ✗ Error inesperado al registrar {img_path.name}: {e}", exc_info=True)
                error_count += 1

    if registered_count > 0 or error_count > 0:
        print(f"\nResumen del registro automático: Registrados: {registered_count}, Errores: {error_count}")
        # Recargar la caché después del registro automático
        try:
            global known_face_encodings, known_face_names
            known_face_encodings, known_face_names = load_known_faces_from_db()
            logger.info(f"Caché de rostros conocidos recargada después del registro automático. Total: {len(known_face_names)}.")
        except Exception as e:
            logger.warning(f"No se pudo recargar la caché de rostros conocidos después del registro automático: {e}")
    else:
        print("\nNo se encontraron nuevas imágenes para registro automático.")


@app.post("/register_face/") # Endpoint renombrado para ser más genérico
async def register_face(
    nombre: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Registra un nuevo rostro subiendo un archivo de imagen.

    - **nombre**: Nombre de la persona
    - **file**: Archivo de imagen que contiene el rostro
    """
    logger.info(f"Recibida solicitud de registro para {nombre} con archivo: {file.filename}")

    # Leer la imagen desde el UploadFile
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error al leer el archivo subido {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo subido: {e}")

    # Procesar la imagen y extraer el embedding
    try:
        rgb_image, process_info = process_image(contents)

        if not process_info.get('success'):
            error_msg = process_info.get('error', 'Error desconocido al procesar la imagen')
            logger.error(f"Error al procesar la imagen subida {file.filename}: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {error_msg}")

        success, face_encoding, face_metadata = extract_face_embedding(rgb_image)

        if not success or face_encoding is None:
            error_msg = face_metadata.get('error', 'No se pudo extraer el rostro')
            logger.warning(f"No se pudo extraer el rostro de la imagen subida {file.filename}: {error_msg}")
            raise HTTPException(status_code=400, detail=f"No se pudo extraer el rostro de la imagen: {error_msg}")

        logger.info(f"Embedding extraído para {nombre} desde {file.filename}.")

    except HTTPException:
        raise # Re-lanzar HTTPExceptions
    except Exception as e:
        logger.error(f"Error inesperado durante el procesamiento de la imagen subida {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la imagen: {e}")

    # Verificar si ya existe un rostro similar (usando la caché actual)
    try:
        is_duplicate, duplicate_name, distance = is_face_duplicate(
            face_encoding,
            known_face_encodings,
            known_face_names
        )

        if is_duplicate:
            logger.warning(f"Intento de registrar a '{nombre}', pero ya existe un rostro similar: '{duplicate_name}' (distancia: {distance:.4f})")
            raise HTTPException(status_code=409, detail=f"Este rostro ya está registrado como '{duplicate_name}'.")

    except Exception as e:
        logger.error(f"Error inesperado al verificar duplicados para {nombre}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al verificar duplicados: {e}")


    # Guardar en la base de datos
    try:
        person_id = save_face_to_db(nombre, face_encoding.tolist())
        logger.info(f"'{nombre}' registrado exitosamente en la DB con ID: {person_id}")

        # Recargar caché después de registrar
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces_from_db()
        logger.info(f"Caché de rostros conocidos recargada después de registrar a '{nombre}'. Total: {len(known_face_names)}.")

        return {
            "message": f"'{nombre}' registrado exitosamente.",
            "person_id": person_id,
            "nombre": nombre
        }

    except Exception as e:
        logger.error(f"Error al guardar '{nombre}' en la base de datos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al registrar: {e}")


@app.post("/recognize_face/")
async def recognize_face(file: UploadFile = File(...)):
    """
    Compara una nueva foto subida con los rostros cargados de la base de datos.
    Retorna el nombre del rostro reconocido o "Desconocido".
    """
    logger.info(f"Recibida solicitud de reconocimiento para archivo: {file.filename}")
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Intento de subir archivo no imagen: {file.content_type}")
        raise HTTPException(status_code=400, detail="Formato de archivo no válido. Se esperaba una imagen.")

    # Leer la imagen desde el UploadFile
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error al leer el archivo subido {file.filename} para reconocimiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo subido para reconocimiento: {e}")

    # Procesar la imagen y extraer el embedding
    try:
        rgb_image, process_info = process_image(contents)

        if not process_info.get('success'):
            error_msg = process_info.get('error', 'Error desconocido al procesar la imagen')
            logger.error(f"Error al procesar la imagen subida {file.filename} para reconocimiento: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Error al procesar la imagen para reconocimiento: {error_msg}")

        success, test_face_encoding, face_metadata = extract_face_embedding(rgb_image)

        if not success or test_face_encoding is None:
            error_msg = face_metadata.get('error', 'No se pudo extraer el rostro')
            logger.warning(f"No se pudo extraer el rostro de la imagen subida {file.filename} para reconocimiento: {error_msg}")
            # Consider returning a specific response for no face detected
            return RecognitionResult(name="No Rostro Detectado", is_known=False, distance=None)

        logger.info(f"Embedding extraído para reconocimiento desde {file.filename}.")

    except HTTPException:
        raise # Re-lanzar HTTPExceptions
    except Exception as e:
        logger.error(f"Error inesperado durante el procesamiento de la imagen subida {file.filename} para reconocimiento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la imagen para reconocimiento: {e}")


    if not known_face_encodings:
        logger.warning("No hay rostros de referencia cargados en memoria para comparación.")
        return RecognitionResult(name="No hay rostros de referencia cargados en memoria.", is_known=False, distance=None)

    # Realizar la comparación (lógica de face_recognition)
    try:
        # Usar la función recognize_face_util de face_utils
        recognition_result = recognize_face_util(test_face_encoding, known_face_encodings, known_face_names)

        if recognition_result.is_known:
            logger.info(f"Rostro reconocido como '{recognition_result.name}' con distancia {recognition_result.distance:.4f}")
        else:
            logger.info(f"Rostro no reconocido (mejor match a distancia {recognition_result.distance:.4f})")

        return RecognitionResult(
            name=recognition_result.name,
            is_known=recognition_result.is_known,
            distance=recognition_result.distance
        )

    except Exception as e:
        logger.error(f"Error durante la comparación de rostros para {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error durante la comparación de rostros: {e}")


@app.delete("/delete_face/{nombre}")
async def delete_face_by_name(nombre: str): # Renombrado para evitar conflicto con la función delete_face_from_db
    """Elimina una persona registrada de la base de datos por su nombre."""
    logger.info(f"Recibida solicitud para eliminar a '{nombre}'")

    try:
        # Buscar el ID por nombre primero
        conn = None
        cur = None
        person_id = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id FROM mis_personas_pg WHERE nombre = %s", (nombre,))
            row = cur.fetchone()
            if row:
                person_id = row[0]
            else:
                 logger.warning(f"Intento de eliminar a '{nombre}', pero no fue encontrado en la DB.")
                 raise HTTPException(status_code=404, detail=f"Persona con nombre '{nombre}' no encontrada.")
        finally:
            if cur: cur.close()
            if conn: conn.close()


        # Llamar a la función de eliminación de la base de datos por ID
        success = delete_face_from_db(person_id) # Usar la función importada

        if success:
            logger.info(f"'{nombre}' (ID: {person_id}) eliminado exitosamente de la base de datos.")
            # Recargar los embeddings en memoria de la API
            try:
                 global known_face_encodings, known_face_names
                 known_face_encodings, known_face_names = load_known_faces_from_db()
                 logger.info("Caché de rostros conocidos recargada después de eliminación.")
            except Exception as e:
                 logger.warning(f"No se pudo recargar la caché de rostros conocidos después de eliminar a '{nombre}': {e}")

            return {"message": f"Persona '{nombre}' eliminada exitosamente de la DB."}
        else:
             # Esto no debería ocurrir si el ID fue encontrado, pero como fallback
             logger.error(f"Fallo inesperado al eliminar a '{nombre}' (ID: {person_id}) de la DB.")
             raise HTTPException(status_code=500, detail=f"Error interno al eliminar a la persona.")

    except HTTPException:
        raise # Re-lanzar HTTPExceptions
    except Exception as e:
        logger.error(f"Error inesperado al eliminar a '{nombre}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno al eliminar: {e}")

@app.get("/list_known_faces/")
async def list_known_faces_endpoint(): # Renombrado para evitar conflicto con la función load_known_faces_from_db
    """Lista todos los nombres de las personas con embeddings cargados."""
    logger.info("Recibida solicitud para listar rostros conocidos.")
    # Devolver los nombres de la caché en memoria
    return {"known_faces": known_face_names}

@app.get("/status/")
async def get_status_endpoint(): # Renombrado para evitar conflicto con la función get_db_status
    """Retorna el estado de la API y el número de rostros cargados."""
    logger.info("Recibida solicitud de estado.")

    # Usar la función get_db_status de app.database
    db_status_info = get_db_status()

    status_info = {
        "status": "running",
        "loaded_faces_count": len(known_face_encodings), # Contar embeddings, no solo nombres
        "source": "Cargado desde la tabla 'mis_personas_pg' de PostgreSQL (embeddings no nulos)",
        "database_connection": db_status_info.get('status', 'Desconocido'),
        "database_details": db_status_info, # Incluir detalles adicionales de la DB
        "message": "Listo para reconocer rostros."
    }
    logger.info(f"Reportando estado: {status_info}")
    return status_info

@app.post("/refresh_db_faces/")
async def refresh_db_faces_endpoint(): # Renombrado para evitar conflicto con la función load_known_faces_from_db
    """Recarga los rostros de referencia desde la base de datos."""
    logger.info("Recibida solicitud para recargar rostros desde la DB.")
    try:
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces_from_db()
        logger.info(f"Rostros recargados desde la DB. Total: {len(known_face_names)}.")
        return {"message": f"Rostros recargados desde la DB. Total: {len(known_face_names)}"}
    except Exception as e:
         logger.error(f"Error al recargar rostros desde la DB: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error interno al recargar rostros: {e}")


if __name__ == "__main__":
    import uvicorn
    # Ejecutar la API usando uvicorn
    logger.info("Iniciando servidor Uvicorn para la API de Reconocimiento Facial.")
    # Asegúrate de usar la ruta correcta para el módulo si estás ejecutando desde la raíz del proyecto
    # uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

    # Si ejecutas este archivo directamente, puedes usar __name__
    uvicorn.run(__name__ + ":app", host="0.0.0.0", port=8000, reload=True)