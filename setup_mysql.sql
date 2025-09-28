-- Script para crear la tabla mis_personas_pg en PostgreSQL (para usar en PGAdmin)

CREATE TABLE IF NOT EXISTS mis_personas_pg (
    id INT AUTO_INCREMENT PRIMARY KEY, -- AUTO_INCREMENT para clave primaria en MySQL
    nombre VARCHAR(255) NOT NULL UNIQUE, -- Nombre de la persona, debe ser único
    embedding FLOAT, -- Puedes usar FLOAT o DOUBLE para almacenar datos numéricos; para arrays, considera usar JSON o una tabla relacionada
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Marca de tiempo de creación
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP -- Marca de tiempo de última actualización
);

-- Nota sobre updated_at: PostgreSQL no actualiza automáticamente la columna TIMESTAMP
-- en cada UPDATE como MySQL. Si necesitas este comportamiento, deberás implementar un TRIGGER.
-- La restricción UNIQUE en 'nombre' ya está definida en la columna.