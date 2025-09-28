import face_recognition
import numpy as np
import cv2
import io
import logging
from typing import List, Optional, Tuple, Any, Dict
from pydantic import BaseModel # Usaremos BaseModel para el resultado de reconocimiento

logger = logging.getLogger(__name__)

# Tolerancia para la comparación de rostros (ajustar según sea necesario, 0.6 es común)
FACE_RECOGNITION_TOLERANCE = 0.6

# Modelo para el resultado de reconocimiento (coherente con main.py)
class FaceRecognitionResult(BaseModel):
    name: str
    is_known: bool
    distance: Optional[float] = None

def process_image(image_data: bytes) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Procesa bytes de imagen para decodificarla y convertirla a formato RGB.

    Args:
        image_data: Bytes de la imagen.

    Returns:
        Tupla de (imagen_rgb, info) donde imagen_rgb es el array NumPy
        en formato RGB o None si falla, e info es un diccionario con detalles.
    """
    info = {'success': False, 'error': None}
    try:
        # face_recognition.load_image_file maneja bytes directamente usando Pillow
        # Esto es más robusto que cv2.imdecode en algunos casos.
        image = face_recognition.load_image_file(io.BytesIO(image_data))
        info['success'] = True
        logger.debug("Imagen cargada y decodificada con face_recognition.load_image_file")
        return image, info
    except Exception as e:
        info['error'] = f"Error al cargar/decodificar la imagen: {str(e)}"
        logger.error(info['error'])
        info['success'] = False
        return None, info

def extract_face_embedding(image_rgb: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
    """
    Extrae el embedding facial de la imagen RGB.

    Args:
        image_rgb: Array NumPy de la imagen en formato RGB.

    Returns:
        Tupla de (éxito, embedding, metadata)
        - éxito: Booleano indicando si se extrajo un embedding.
        - embedding: Array NumPy del embedding facial (128 dimensiones) o None.
        - metadata: Diccionario con información adicional (ej. ubicación del rostro, errores).
    """
    metadata = {'face_locations': [], 'error': None}
    try:
        # Encontrar todos los rostros en la imagen
        face_locations = face_recognition.face_locations(image_rgb)
        metadata['face_locations'] = face_locations
        logger.debug(f"Detectados {len(face_locations)} rostro(s).")

        if not face_locations:
            metadata['error'] = "No se encontraron rostros en la imagen."
            logger.warning(metadata['error'])
            return False, None, metadata

        # Asumimos que procesamos el primer rostro encontrado
        # Puedes modificar esto si necesitas manejar múltiples rostros
        face_encoding = face_recognition.face_encodings(image_rgb, [face_locations[0]])[0]
        logger.debug("Embedding facial extraído exitosamente.")
        return True, face_encoding, metadata

    except Exception as e:
        metadata['error'] = f"Error durante la detección o extracción del embedding: {str(e)}"
        logger.error(metadata['error'])
        return False, None, metadata

def recognize_face(
    test_embedding: np.ndarray,
    known_face_encodings: List[np.ndarray],
    known_face_names: List[str],
    tolerance: float = FACE_RECOGNITION_TOLERANCE
) -> FaceRecognitionResult:
    """
    Compara un embedding de prueba con una lista de embeddings conocidos.

    Args:
        test_embedding: Embedding facial a reconocer.
        known_face_encodings: Lista de embeddings faciales conocidos (NumPy arrays).
        known_face_names: Lista de nombres correspondientes a los embeddings conocidos.
        tolerance: Umbral de distancia para considerar una coincidencia.

    Returns:
        Un objeto FaceRecognitionResult con el resultado.
    """
    if not known_face_encodings:
        logger.warning("Lista de rostros conocidos vacía para comparación.")
        return FaceRecognitionResult(name="No hay rostros de referencia cargados en memoria.", is_known=False, distance=None)

    # Calcular distancias a todos los rostros conocidos
    face_distances = face_recognition.face_distance(known_face_encodings, test_embedding)

    # Encontrar el mejor match (el de menor distancia)
    best_match_index = np.argmin(face_distances)
    best_match_distance = face_distances[best_match_index]

    # Verificar si el mejor match está dentro del umbral de tolerancia
    if best_match_distance <= tolerance:
        name = known_face_names[best_match_index]
        logger.debug(f"Rostro reconocido como '{name}' con distancia {best_match_distance:.4f}")
        return FaceRecognitionResult(name=name, is_known=True, distance=float(best_match_distance))
    else:
        logger.debug(f"Rostro no reconocido (mejor match a distancia {best_match_distance:.4f})")
        return FaceRecognitionResult(name="Desconocido", is_known=False, distance=float(best_match_distance))

def is_face_duplicate(
    new_embedding: np.ndarray,
    known_face_encodings: List[np.ndarray],
    known_face_names: List[str],
    tolerance: float = FACE_RECOGNITION_TOLERANCE
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Verifica si un nuevo embedding facial ya existe en la lista de rostros conocidos
    basado en una tolerancia.

    Args:
        new_embedding: El embedding facial a verificar.
        known_face_encodings: Lista de embeddings faciales conocidos (NumPy arrays).
        known_face_names: Lista de nombres correspondientes a los embeddings conocidos.
        tolerance: Umbral de distancia para considerar un duplicado.

    Returns:
        Tupla de (es_duplicado, nombre_del_duplicado, distancia_al_duplicado)
    """
    if not known_face_encodings:
        logger.debug("Lista de rostros conocidos vacía, no hay duplicados posibles.")
        return False, None, None

    # Calcular distancias a todos los rostros conocidos
    face_distances = face_recognition.face_distance(known_face_encodings, new_embedding)

    # Encontrar el mejor match
    best_match_index = np.argmin(face_distances)
    best_match_distance = face_distances[best_match_index]

    # Si el mejor match está dentro del umbral de tolerancia, es un duplicado
    if best_match_distance <= tolerance:
        duplicate_name = known_face_names[best_match_index]
        logger.warning(f"Detectado posible duplicado: '{duplicate_name}' con distancia {best_match_distance:.4f}")
        return True, duplicate_name, float(best_match_distance)
    else:
        logger.debug(f"No se encontró duplicado dentro de la tolerancia {tolerance} (mejor match a distancia {best_match_distance:.4f}).")
        return False, None, None