import psycopg2
from dotenv import load_dotenv
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
# Aunque usamos valores hardcodeados para este test, es buena práctica mantener esto
# si el script va a ser usado en otros contextos que sí usen .env
load_dotenv()

def check_database():
    """Verifica la conexión a la base de datos PostgreSQL y la estructura básica de la tabla usando valores especificados."""
    logger.info("Intentando conectar a la base de datos PostgreSQL con los valores especificados...")

    # Obtener configuración de la base de datos - usando valores hardcodeados para esta prueba
    db_config = {
        'host': 'localhost',
        'database': 'reconocimiento', # Nombre de base de datos
        'user': 'Postgres',
        'password': 'admin',
        'port': '5433'
    }

    # Registrar configuración (sin la contraseña por seguridad)
    logger.info(f"Configuración DB - Host: {db_config['host']}, Base de Datos: {db_config['database']}, Usuario: {db_config['user']}, Puerto: {db_config['port']}")

    connection = None
    try:
        # Intentar conectar a la base de datos
        connection = psycopg2.connect(**db_config)

        if connection.isolation_level is not None: # Verificar si la conexión está activa
            logger.info("✅ Conexión exitosa a la base de datos PostgreSQL.")

            # Obtener el nombre de la base de datos actual
            cursor = connection.cursor()
            cursor.execute("SELECT current_database();")
            db_name = cursor.fetchone()[0]
            logger.info(f"Conectado a la base de datos: {db_name}")

            # Verificar si la tabla 'mis_personas_pg' existe (coherente con el script de setup PG)
            table_name = 'mis_personas_pg'
            cursor.execute("""
                SELECT EXISTS (
                   SELECT 1
                   FROM information_schema.tables
                   WHERE table_schema = current_schema() -- Asumiendo esquema por defecto
                   AND table_name = %s
                );
            """, (table_name,))
            table_exists = cursor.fetchone()[0]

            if table_exists:
                logger.info(f"✅ La tabla '{table_name}' existe.")

                # Contar registros
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"Total de registros en '{table_name}': {count}")

                # Opcional: Verificar estructura de la tabla (puede ser detallado)
                # logger.info(f"Estructura de la tabla '{table_name}':")
                # cursor.execute(f"""
                #     SELECT column_name, data_type
                #     FROM information_schema.columns
                #     WHERE table_name = '{table_name}'
                # """)
                # for col in cursor.fetchall():
                #     logger.info(f"- {col[0]}: {col[1]}")

            else:
                logger.warning(f"❌ La tabla '{table_name}' NO existe.")
                # Podrías querer lanzar un error o retornar False aquí
                return False # Indicar fallo si la tabla crítica falta

            cursor.close()
            return True # Indicar éxito

    except psycopg2.Error as e:
        logger.error(f"❌ Error al conectar a PostgreSQL o ejecutar la consulta: {e}")
        return False

    except Exception as e:
        logger.error(f"❌ Ocurrió un error inesperado: {e}", exc_info=True)
        return False

    finally:
        if connection:
            connection.close()
            logger.info("La conexión a PostgreSQL está cerrada.")

if __name__ == "__main__":
    logger.info("Iniciando script de verificación de base de datos...")
    if check_database():
        logger.info("Verificación de base de datos completada exitosamente.")
    else:
        logger.error("Verificación de base de datos fallida.")