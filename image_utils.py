"""
Funciones de utilidad para carga y procesamiento robusto de imágenes.
"""
import os
import cv2
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def load_image_with_retry(
    image_path: Union[str, Path],
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    convert_to_rgb: bool = True,
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Carga una imagen con mecanismo de reintentos e informe detallado de errores.

    Args:
        image_path: Ruta al archivo de imagen
        max_attempts: Número máximo de intentos de carga
        initial_delay: Retraso inicial entre reintentos en segundos
        backoff_factor: Multiplicador para el retraso entre reintentos
        convert_to_rgb: Si convertir de BGR a RGB
        **kwargs: Argumentos adicionales para cv2.imread()

    Returns:
        Tupla de (imagen, dict_info) donde:
        - imagen: Array de la imagen cargada o None si todos los intentos fallaron
        - dict_info: Diccionario con información de depuración y detalles del error
    """
    image_path = Path(image_path) if not isinstance(image_path, Path) else image_path
    info = {
        'success': False,
        'attempts': 0,
        'errors': [],
        'file_size': None,
        'file_permissions': None,
        'file_exists': None,
        'file_path': str(image_path.absolute()),
        'file_extension': image_path.suffix.lower(),
        'image_shape': None,
        'image_dtype': None,
        'processing_time': None
    }

    # Verificar primero la existencia y permisos del archivo
    if not image_path.exists():
        error_msg = f"El archivo no existe: {image_path}"
        logger.error(error_msg)
        info['errors'].append(error_msg)
        return None, info

    info['file_exists'] = True
    info['file_size'] = os.path.getsize(image_path)

    try:
        info['file_permissions'] = oct(os.stat(image_path).st_mode)[-3:]
    except Exception as e:
        info['file_permissions'] = f"Error: {str(e)}"

    # Verificar tamaño del archivo
    if info['file_size'] == 0:
        error_msg = f"El archivo está vacío: {image_path}"
        logger.error(error_msg)
        info['errors'].append(error_msg)
        return None, info

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        info['attempts'] = attempt
        start_time = time.time()

        try:
            # Intentar leer la imagen
            image = cv2.imread(str(image_path), **kwargs)

            if image is None:
                error_msg = f"Intento {attempt}: Falló la decodificación de la imagen (cv2.imread devolvió None)"
                logger.warning(error_msg)
                info['errors'].append(error_msg)

                # Intentar método de lectura alternativo en el último intento
                if attempt == max_attempts:
                    try:
                        with open(image_path, 'rb') as f:
                            image_data = np.frombuffer(f.read(), np.uint8)
                        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                        if image is not None:
                            logger.info("Imagen leída exitosamente usando método alternativo")
                    except Exception as e:
                        error_msg = f"Falló el método de lectura alternativo: {str(e)}"
                        logger.error(error_msg)
                        info['errors'].append(error_msg)

            # Si tenemos una imagen válida
            if image is not None:
                info['success'] = True
                info['image_shape'] = image.shape
                info['image_dtype'] = str(image.dtype)

                if convert_to_rgb and len(image.shape) >= 3:  # Solo convertir si es una imagen a color
                    try:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        info['color_space'] = 'RGB'
                    except Exception as e:
                        error_msg = f"Falló la conversión de color: {str(e)}"
                        logger.warning(error_msg)
                        info['errors'].append(error_msg)

                info['processing_time'] = time.time() - start_time
                logger.info(f"Imagen cargada exitosamente: {image_path}")
                return image, info

        except Exception as e:
            error_msg = f"Intento {attempt}: Error inesperado: {str(e)}"
            logger.error(error_msg, exc_info=True)
            info['errors'].append(error_msg)

        # Si este intento falló
        if attempt < max_attempts:
            logger.info(f"Reintentando en {delay:.2f} segundos... (Intento {attempt + 1}/{max_attempts})")
            time.sleep(delay)
            delay *= backoff_factor

    # Si todos los intentos fallaron
    info['processing_time'] = time.time() - start_time if 'start_time' in locals() else None
    logger.error(f"Falló la carga de la imagen después de {max_attempts} intentos: {image_path}")
    return None, info

def check_image_quality(
    image: np.ndarray,
    min_width: int = 50,
    min_height: int = 50,
    min_face_size: int = 30
) -> Dict[str, Any]:
    """
    Verifica la calidad de la imagen y detecta posibles problemas.

    Args:
        image: Imagen de entrada
        min_width: Ancho mínimo aceptable en píxeles
        min_height: Alto mínimo aceptable en píxeles
        min_face_size: Tamaño mínimo de rostro a considerar válido

    Returns:
        Diccionario con métricas de calidad y problemas detectados
    """
    result = {
        'width': image.shape[1],
        'height': image.shape[0],
        'channels': image.shape[2] if len(image.shape) > 2 else 1,
        'is_too_small': False,
        'is_too_large': False,
        'is_low_contrast': False,
        'is_blurry': False,
        'has_faces': False,
        'face_count': 0,
        'face_sizes': [],
        'issues': []
    }

    # Verificar dimensiones de la imagen
    if result['width'] < min_width or result['height'] < min_height:
        result['is_too_small'] = True
        result['issues'].append(f"La imagen es demasiado pequeña ({result['width']}x{result['height']})")

    # Verificar si la imagen es demasiado grande (opcional)
    if result['width'] > 8000 or result['height'] > 8000:
        result['is_too_large'] = True
        result['issues'].append(f"La imagen es muy grande ({result['width']}x{result['height']})")

    # Convertir a escala de grises para verificaciones de calidad
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Verificar contraste (usando desviación estándar de intensidades de píxeles)
    contrast = np.std(gray)
    if contrast < 20:  # Umbral para bajo contraste
        result['is_low_contrast'] = True
        result['issues'].append(f"Bajo contraste (desviación estándar: {contrast:.1f})")

    # Verificar desenfoque (usando varianza Laplaciana)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 50:  # Umbral para imagen desenfocada
        result['is_blurry'] = True
        result['issues'].append(f"La imagen está desenfocada (varianza Laplaciana: {blur:.1f})")

    # Intentar detectar rostros si la imagen es lo suficientemente grande
    if result['width'] >= min_face_size and result['height'] >= min_face_size:
        try:
            # Usar un detector de rostros más eficiente para esta verificación
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size)
            )

            result['face_count'] = len(faces)
            result['has_faces'] = len(faces) > 0
            result['face_sizes'] = [w * h for (x, y, w, h) in faces]

            if not result['has_faces']:
                result['issues'].append("No se detectaron rostros")

        except Exception as e:
            logger.warning(f"Falló la detección de rostros: {str(e)}")
            result['issues'].append("Falló la detección de rostros")

    return result