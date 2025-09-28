import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def cleanup_orphaned_photos(image_folder: Path, database_names: list):
    """
    Elimina fotos en la carpeta de imágenes que no tienen una entrada correspondiente
    en la lista de nombres de la base de datos.

    Args:
        image_folder: Ruta al directorio que contiene las imágenes.
        database_names: Lista de nombres de personas registrados en la base de datos.
    """
    logger.info(f"Iniciando limpieza de fotos huérfanas en: {image_folder}")
    if not image_folder.is_dir():
        logger.warning(f"Directorio de imágenes no encontrado para limpieza: {image_folder}")
        return

    cleaned_count = 0
    for img_path in image_folder.glob('*'):
        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_name = img_path.stem # Nombre del archivo sin extensión

            if image_name not in database_names:
                try:
                    os.remove(img_path)
                    logger.info(f"Eliminada foto huérfana: {img_path.name}")
                    cleaned_count += 1
                except OSError as e:
                    logger.error(f"Error al eliminar foto huérfana {img_path.name}: {e}")

    logger.info(f"Limpieza de fotos huérfanas completada. Fotos eliminadas: {cleaned_count}")
