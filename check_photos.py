import os
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from dotenv import load_dotenv
import logging
import json
from datetime import datetime
import argparse # Importar argparse aquí

# Importar nuestras nuevas funciones de utilidad
from image_utils import load_image_with_retry, check_image_quality

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('photo_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_directories(single_image_path=None):
    """Configura y verifica los directorios necesarios o procesa una única imagen."""
    try:
        # Cargar variables de entorno
        load_dotenv()

        # Obtener directorio base
        base_dir = Path(__file__).resolve().parent
        logger.info(f"Directorio base: {base_dir}")

        # Si se proporciona una ruta de imagen única, usarla directamente
        if single_image_path:
            single_image_path = Path(single_image_path)
            if not single_image_path.is_absolute():
                single_image_path = base_dir / single_image_path

            if not single_image_path.exists():
                logger.error(f"Archivo de imagen no encontrado: {single_image_path}")
                return base_dir, None, None

            logger.info(f"Procesando imagen única: {single_image_path}")
            # Crear directorio de resultados en el directorio base para imagen única también
            results_dir = base_dir / "analysis_results"
            results_dir.mkdir(exist_ok=True)
            logger.info(f"Los resultados se guardarán en: {results_dir}")
            return base_dir, single_image_path, results_dir

        # De lo contrario, usar el directorio de fotos de referencia por defecto
        image_folder = base_dir / "reference_photos"
        logger.info(f"Buscando imágenes en: {image_folder}")

        # Crear directorio de resultados
        results_dir = base_dir / "analysis_results"
        results_dir.mkdir(exist_ok=True)
        logger.info(f"Los resultados se guardarán en: {results_dir}")

        # Verificar existencia del directorio de imágenes
        if not image_folder.exists():
            # Lógica interactiva para encontrar o crear el directorio... (simplificada para concisión)
            error_msg = f"Error: Directorio '{image_folder.name}' no encontrado en {base_dir}"
            logger.error(error_msg)
            # En un script real, aquí podría haber una pausa e interacción
            raise FileNotFoundError(error_msg) # Simplificado: lanzar error en lugar de interactuar

        logger.info(f"Usando directorio de imágenes: {image_folder}")
        return base_dir, image_folder, results_dir

    except Exception as e:
        logger.exception("Error en setup_directories():")
        raise

def find_image_files(image_folder: Path) -> list:
    """Encuentra todos los archivos de imagen válidos en el directorio especificado."""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.jfif', '.webp']
    image_files = [f for f in image_folder.glob('*')
                  if f.suffix.lower() in valid_extensions and f.is_file()] # Asegurar que sea un archivo

    if not image_files:
        error_msg = f"No se encontraron archivos de imagen {', '.join(valid_extensions)} en el directorio: {image_folder}"
        logger.warning(error_msg) # Cambiado a warning ya que el script puede continuar sin imágenes
        # No lanzar FileNotFoundError aquí si el modo es de directorio y solo no hay archivos
        return [] # Retornar lista vacía si no hay archivos


    return sorted(image_files)

def process_face_detection(image: np.ndarray, min_face_size: int = 30) -> dict:
    """Procesa la detección de rostros en la imagen dada."""
    result = {
        'face_count': 0,
        'face_locations': [],
        'face_encodings': [],
        'face_sizes': [],
        'errors': [] # Lista de errores específicos de detección/encoding
    }

    try:
        # Convertir a RGB si es necesario (face_recognition usa RGB)
        # Mejorar la conversión para manejar imágenes en escala de grises directamente
        if len(image.shape) == 2: # Si es escala de grises
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3: # Si es color
             # Asegurarse de que sea BGR a RGB, no RGB a RGB accidentalmente
            if image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4: # Manejar RGBA
                 rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                 result['errors'].append(f"Formato de color inesperado: {image.shape[2]} canales")
                 logger.warning(result['errors'][-1])
                 return result # Salir si el formato es inesperado
        else:
            result['errors'].append(f"Formato de imagen inesperado: {image.shape}")
            logger.warning(result['errors'][-1])
            return result # Salir si el formato es inesperado


        # Detectar rostros
        face_locations = face_recognition.face_locations(rgb_image)
        result['face_count'] = len(face_locations)

        if not face_locations:
            # No es un error crítico si no hay rostros, solo se reporta
            return result

        # Procesar cada rostro
        for face_location in face_locations:
            top, right, bottom, left = face_location
            # Asegurar que las dimensiones sean positivas antes de calcular el tamaño
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                 error_msg = f"Dimensiones de rostro no válidas: w={width}, h={height}"
                 result['errors'].append(error_msg)
                 logger.warning(error_msg)
                 continue # Saltar este rostro inválido

            face_size = width * height

            if face_size < min_face_size * min_face_size:
                result['errors'].append(f"Rostro demasiado pequeño: {face_size}px²")
                continue # Saltar rostros pequeños

            result['face_sizes'].append(face_size)
            result['face_locations'].append(face_location)

            # Obtener encodings del rostro
            try:
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    [face_location],
                    model="large" # Usar el modelo más grande para mejor precisión
                )
                if encodings:
                    result['face_encodings'].append(encodings[0].tolist())
                else:
                     error_msg = "Face encoding falló (lista vacía)"
                     logger.warning(error_msg)
                     result['errors'].append(error_msg)

            except Exception as e:
                error_msg = f"Face encoding falló: {str(e)}"
                logger.warning(error_msg)
                result['errors'].append(error_msg)

    except Exception as e:
        # Capturar errores generales del proceso de detección
        error_msg = f"Error general en la detección de rostros: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result['errors'].append(error_msg)

    return result

def analyze_image(image_path: Path) -> dict:
    """Analiza una única imagen con manejo detallado de errores y calidad."""
    result = {
        'file_name': image_path.name,
        'file_path': str(image_path.absolute()),
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'load_info': {}, # Información de carga (añadida por load_image_with_retry)
        'quality': {}, # Información de calidad (añadida por check_image_quality)
        'face_info': {}, # Información de detección de rostros (añadida por process_face_detection)
        'warnings': [], # Advertencias generales (ej: no hay rostros, baja calidad)
        'errors': [] # Errores críticos que impiden el análisis completo
    }

    logger.info(f"\n{'='*80}\nProcesando: {image_path.name}")

    # Paso 1: Cargar la imagen con reintentos
    image, load_info = load_image_with_retry(
        image_path,
        max_attempts=3,
        initial_delay=0.5,
        backoff_factor=2.0
        # No convertimos a RGB aquí si check_image_quality o process_face_detection lo necesitan,
        # lo harán internamente para evitar conversiones dobles.
        # Se ha modificado load_image_with_retry para permitir esto.
    )

    # Actualizar resultado con información de carga
    result['load_info'] = load_info
    if not load_info['success']:
        error_msg = f"Fallo al cargar la imagen después de {load_info['attempts']} intentos"
        logger.error(error_msg)
        result['errors'].append(error_msg)
        result['errors'].extend(load_info.get('errors', [])) # Añadir errores específicos de carga
        return result # Salir si la carga falla

    # Paso 2: Verificar calidad de la imagen
    # Asegurarse de que la imagen cargada no sea None antes de pasarla a check_image_quality
    if image is not None:
        quality_info = check_image_quality(image)
        result['quality'] = quality_info

        if quality_info.get('issues'):
            logger.warning(f"Problemas de calidad: {', '.join(quality_info['issues'])}")
            result['warnings'].extend(quality_info['issues'])
    else:
         # Esto no debería ocurrir si load_info['success'] es True, pero como fallback
         error_msg = "Imagen cargada es None después de un intento 'exitoso'."
         logger.error(error_msg)
         result['errors'].append(error_msg)
         return result


    # Paso 3: Procesar detección de rostros
    # Asegurarse de que la imagen cargada no sea None antes de pasarla a process_face_detection
    if image is not None:
        face_info = process_face_detection(image)
        result['face_info'] = face_info

        if face_info.get('face_count', 0) > 0:
            logger.info(f"Detectados {face_info['face_count']} rostro(s) en la imagen")
            if face_info.get('face_sizes'):
                logger.info(f"Tamaños de rostros: {', '.join(map(str, face_info['face_sizes']))}px²")
        else:
            warning_msg = "No se detectaron rostros en la imagen"
            logger.warning(warning_msg)
            result['warnings'].append(warning_msg)

        # Añadir errores específicos de detección/encoding a los errores generales
        if face_info.get('errors'):
             result['errors'].extend(face_info['errors'])


    # Marcar éxito si no hay errores críticos (advertencias están bien)
    result['success'] = not result['errors'] # Éxito si la lista de errores está vacía

    logger.info(f"Análisis completado para {image_path.name}. Éxito: {result['success']}")
    return result

def save_results(results: list, output_dir: Path):
    """Guarda los resultados del análisis en un archivo JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"photo_analysis_summary_{timestamp}.json"

    try:
        # Calcular resumen
        total_images = len(results)
        successful_analyses = sum(1 for r in results if r.get('success', False))
        failed_analyses = total_images - successful_analyses
        images_with_faces = sum(1 for r in results if r.get('face_info', {}).get('face_count', 0) > 0)
        images_without_faces = total_images - images_with_faces
        images_with_quality_issues = sum(1 for r in results if r.get('quality', {}).get('issues'))
        images_with_errors = sum(1 for r in results if r.get('errors'))


        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_images_processed': total_images,
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'images_with_faces_detected': images_with_faces,
            'images_without_faces_detected': images_without_faces,
            'images_with_quality_issues': images_with_quality_issues,
            'images_with_processing_errors': images_with_errors,
            'results_list': results # Incluir la lista detallada de resultados
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Resultados guardados en {output_file}")
        return output_file
    except Exception as e:
        error_msg = f"Fallo al guardar los resultados: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

def main():
    """Función principal para ejecutar el análisis de fotos."""
    try:
        # Parsear argumentos de línea de comandos
        parser = argparse.ArgumentParser(description='Analiza fotos para detección de rostros y calidad.')
        # Usar un grupo de argumentos mutuamente excluyentes
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--image', '-i', help='Procesa un único archivo de imagen.')
        group.add_argument('--directory', '-d', help='Procesa todas las imágenes en un directorio.')
        args = parser.parse_args()

        # Setup basado en el tipo de entrada
        if args.image:
            # Modo imagen única
            base_dir, image_path, results_dir = setup_directories(args.image)
            if not image_path: # Si setup_directories falló o la imagen no existe
                return 1 # Salir con código de error

            # Analizar la imagen única
            results = [analyze_image(image_path)]
            output_file = save_results(results, results_dir)
            print(f"\n¡Análisis completo! Resultados guardados en: {output_file}")
            # Retornar 0 si el análisis fue exitoso, 1 si falló (basado en el único resultado)
            return 0 if results[0].get('success', False) else 1

        elif args.directory:
            # Modo directorio
            input_dir = Path(args.directory)
            if not input_dir.is_dir():
                logger.error(f"Directorio no encontrado: {args.directory}")
                return 1

            # Crear directorio de resultados con marca de tiempo para este análisis de directorio
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Asegurarse de que results_dir esté en el mismo nivel que input_dir o en un lugar definido
            # Aquí lo pondremos junto al directorio de entrada para evitar mezclas
            results_dir = input_dir.parent / f"analysis_results_{timestamp}"
            results_dir.mkdir(exist_ok=True)


            # Encontrar todos los archivos de imagen
            image_files = find_image_files(input_dir)
            if not image_files:
                logger.warning("No se encontraron archivos de imagen válidos en el directorio especificado.")
                # Dependiendo del requisito, podríamos salir o reportar 0 imágenes procesadas
                # Aquí, saldremos ya que no hay nada que analizar.
                # También crear el archivo de resultados vacío para documentar que no se encontró nada
                save_results([], results_dir)
                logger.info("Se creó un archivo de resumen vacío.")
                return 1 # Salir con código de error

            logger.info(f"Encontrados {len(image_files)} imágenes en {input_dir}")

            # Procesar cada imagen
            results = []
            for i, img_path in enumerate(image_files, 1):
                try:
                    logger.info(f"\nProcesando imagen {i}/{len(image_files)}: {img_path.name}")
                    result = analyze_image(img_path)
                    results.append(result)
                    logger.info(f"Completado: {img_path.name} - {'Éxito' if result.get('success') else 'Fallo'}")
                except Exception as e:
                    error_msg = f"Error CRÍTICO procesando {img_path.name}: {str(e)}"
                    logger.critical(error_msg, exc_info=True) # Usar critical para errores inesperados
                    results.append({
                        'file_name': img_path.name,
                        'file_path': str(img_path.absolute()),
                        'timestamp': datetime.now().isoformat(),
                        'success': False,
                        'errors': [error_msg],
                        'load_info': {'success': False, 'errors': [error_msg]}, # Simular estructura de error
                        'quality': {},
                        'face_info': {}
                    })

            # Guardar resultados
            output_file = save_results(results, results_dir)

            # Imprimir resumen
            successful_count = sum(1 for r in results if r.get('success', False))
            total_count = len(results)
            logger.info(f"\n{'='*40}\n"
                       f"Análisis completo!\n"
                       f"Imágenes totales procesadas: {total_count}\n"
                       f"Análisis exitosos: {successful_count}\n"
                       f"Análisis fallidos: {total_count - successful_count}\n"
                       f"Resultados guardados en: {output_file}\n"
                       f"{'='*40}")

            # Retornar código de salida: 0 si al menos una imagen se procesó con éxito, 1 si todas fallaron
            return 0 if successful_count > 0 else 1


    except Exception as e:
        # Capturar cualquier error no manejado en main
        logger.critical(f"Error crítico inesperado en main(): {str(e)}", exc_info=True)
        return 1 # Salir con código de error

if __name__ == "__main__":
    # Ejecutar la función principal y usar su código de salida
    exit(main())