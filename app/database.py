import psycopg2
import os
import logging
from typing import List, Tuple, Any, Optional
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de la Base de Datos ---
# Usando la configuración especificada por el usuario
DB_CONFIG = {
    'host': 'localhost',
    'database': 'reconocimiento', # Nombre de base de datos
    'user': 'Postgres',
    'password': 'admin', # Considera usar variables de entorno para la contraseña en producción
    'port': '5433' # Puerto de la base de datos
}

def get_db_connection():
    """Establece y retorna una conexión a la base de datos PostgreSQL."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Conexión a DB establecida exitosamente.")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        # En una API real, podrías querer relanzar una excepción HTTP o manejarla de otra manera
        raise

def get_db_status():
    """Verifica el estado de la conexión a la base de datos."""
    status_info = {
        'status': 'Desconectado',
        'details': 'No se pudo conectar'
    }
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        if conn.isolation_level is not None: # Verificar si la conexión está activa
            cur = conn.cursor()
            cur.execute("SELECT 1;") # Consulta simple para verificar la conexión
            status_info['status'] = 'Conectado'
            status_info['details'] = 'Verificación exitosa'
            logger.info("Verificación de estado de DB exitosa.")
        else:
            status_info['details'] = 'Conexión inactiva'

    except Exception as e:
        status_info['details'] = f"Error durante la verificación: {e}"
        logger.error(f"Error durante la verificación de estado de DB: {e}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return status_info


def load_known_faces_from_db() -> Tuple[List[np.ndarray], List[str]]:
    """
    Carga los embeddings faciales y nombres desde la base de datos
    donde el embedding no es nulo.
    """
    known_face_encodings: List[np.ndarray] = []
    known_face_names: List[str] = []
    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Seleccionar nombre y embedding de la tabla donde el embedding no es NULL
        cur.execute("SELECT nombre, embedding FROM mis_personas_pg WHERE embedding IS NOT NULL;")
        rows = cur.fetchall()

        for nombre, embedding_data in rows:
            # Asegurarse de que embedding_data es una lista antes de convertir a np.array
            if isinstance(embedding_data, list):
                known_face_names.append(nombre)
                known_face_encodings.append(np.array(embedding_data))
            else:
                logger.warning(f"Los datos del embedding para '{nombre}' no son una lista válida. Saltando registro.")


        logger.info(f"Cargados {len(known_face_names)} rostros con embeddings desde la DB.")

    except psycopg2.Error as e:
        logger.error(f"Error de base de datos al cargar rostros: {e}")
        # En una API real, podrías querer relanzar una excepción HTTP o manejarla de otra manera
        raise
    except Exception as e:
        logger.error(f"Error inesperado al cargar rostros: {e}")
        raise

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return known_face_encodings, known_face_names

def save_face_to_db(nombre: str, embedding: List[float]) -> int:
    """
    Guarda un nuevo rostro o actualiza uno existente en la base de datos.
    Retorna el ID del registro guardado/actualizado.
    """
    conn = None
    cur = None
    person_id = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Verificar si el nombre ya existe
        cur.execute("SELECT id FROM mis_personas_pg WHERE nombre = %s;", (nombre,))
        row = cur.fetchone()

        if row:
            # Si existe, actualizar el embedding y updated_at
            person_id = row[0]
            cur.execute(
                "UPDATE mis_personas_pg SET embedding = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s;",
                (embedding, person_id)
            )
            logger.info(f"Rostro para '{nombre}' (ID: {person_id}) actualizado en la DB.")
        else:
            # Si no existe, insertar un nuevo registro
            cur.execute(
                "INSERT INTO mis_personas_pg (nombre, embedding) VALUES (%s, %s) RETURNING id;",
                (nombre, embedding)
            )
            person_id = cur.fetchone()[0] # Obtener el ID del nuevo registro
            logger.info(f"Nuevo rostro para '{nombre}' (ID: {person_id}) insertado en la DB.")

        conn.commit()
        return person_id

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Error de base de datos al guardar rostro para '{nombre}': {e}")
        raise # Re-lanzar el error para que sea manejado por el llamador
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error inesperado al guardar rostro para '{nombre}': {e}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def delete_face_from_db(person_id: int) -> bool:
    """
    Elimina un registro de rostro de la base de datos por su ID.
    Retorna True si se eliminó un registro, False si no se encontró el ID.
    """
    conn = None
    cur = None
    success = False
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("DELETE FROM mis_personas_pg WHERE id = %s;", (person_id,))
        conn.commit()

        if cur.rowcount > 0:
            logger.info(f"Registro con ID {person_id} eliminado de la DB.")
            success = True
        else:
            logger.warning(f"Intento de eliminar registro con ID {person_id}, pero no fue encontrado en la DB.")
            success = False # No se encontró el registro con ese ID

        return success

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        logger.error(f"Error de base de datos al eliminar registro con ID {person_id}: {e}")
        raise # Re-lanzar el error
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error inesperado al eliminar registro con ID {person_id}: {e}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# Nota: get_face_embeddings era importado en el main.py anterior,
# pero load_known_faces_from_db hace una función similar.
# Si get_face_embeddings tiene una función diferente (ej. retornar todos los campos incluyendo ID),
# puedes añadirla aquí. Por ahora, load_known_faces_from_db es suficiente para la caché en memoria.
# Si la necesitas, podría ser algo como:
# def get_face_embeddings() -> List[Tuple[int, str, List[float]]]:
#    conn = None
#    cur = None
#    try:
#        conn = get_db_connection()
#        cur = conn.cursor()
#        cur.execute("SELECT id, nombre, embedding FROM mis_personas_pg;")
#        rows = cur.fetchall()
#        # Asegurarse de que el embedding sea una lista de floats
#        return [(row[0], row[1], list(row[2])) for row in rows if row[2] is not None]
#    except Exception as e:
#        logger.error(f"Error al obtener todos los embeddings: {e}")
#        raise
#    finally:
#        if cur: cur.close()
#        if conn: conn.close()