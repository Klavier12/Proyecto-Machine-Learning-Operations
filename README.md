# MLOps-P1: Proyecto Machine Learning Operations

El proyecto se centra en el desarrollo de una plataforma integral de análisis de datos y recomendaciones para una plataforma de entretenimiento. Se ha llevado a cabo un análisis detallado de los datos mediante el proceso ETL para garantizar la calidad y la integridad de los datos, seguido de un análisis exploratorio (EDA) para comprender mejor las tendencias y patrones. Este análisis ha proporcionado información crucial sobre el contenido de la plataforma y el comportamiento de los usuarios, fundamentando el desarrollo de las funciones de machine learning y el sistema de recomendaciones.

## Tabla de Contenidos

- [Introducción](#introducción)
- [Instalación](#instalación)
- [Uso](#uso)
- [Características](#características)
- [Contacto](#contacto)

## Introducción

En la era digital actual, el análisis de datos y el aprendizaje automático (machine learning) han emergido como pilares fundamentales para la toma de decisiones informadas y el desarrollo de sistemas inteligentes. Desde la detección de patrones hasta la predicción de resultados, las aplicaciones del machine learning abarcan una amplia gama de campos, desde la medicina hasta la industria del entretenimiento.

Este proyecto fusiona la capacidad de realizar operaciones de machine learning con la funcionalidad de un sistema de recomendaciones, ofreciendo una solución versátil para diversas aplicaciones en la industria de los videojuegos.

## Instalación

Para el proyecto se ha utilizado herramientas como Python, librerías como Pandas y NumPy, y archivos de comprensión de datos en formato `.parquet`. Todo esto con el objetivo de realizar de la mejor manera y de forma eficiente la importación de los datos y su manejo para obtener los resultados.

### Fuente de datos

- **Dataset**: Carpeta con los archivos que requieren ser procesados; tengan en cuenta que hay datos que están anidados (un diccionario o una lista como valores en la fila).
- **Diccionario de datos**: Incluye descripciones de las columnas disponibles en el dataset.
- Se han creado y extraído archivos pequeños para realizar un trabajo más específico y eficiente, los cuales se encuentran en la carpeta Dataset.

## Análisis Realizado por ETL y EDA:

1. **Análisis ETL (Extract, Transform, Load)**:
   - Se realizó un proceso exhaustivo de ETL para extraer datos de múltiples fuentes, transformarlos en un formato adecuado y cargarlos en una base de datos centralizada.
   - Las etapas de extracción, transformación y carga se llevaron a cabo con el objetivo de garantizar la integridad, consistencia y calidad de los datos.
   - Se implementaron técnicas de limpieza y preprocesamiento de datos para abordar problemas como valores faltantes, duplicados y formatos inconsistentes.

2. **Análisis EDA (Exploratory Data Analysis)**:
   - Se realizó un análisis exploratorio de los datos para comprender mejor su estructura, distribución y características.
   - Se utilizaron visualizaciones y estadísticas descriptivas para identificar patrones, tendencias y relaciones en los datos.
   - Se llevaron a cabo análisis específicos para explorar la distribución de contenido por año, el comportamiento de compra de los usuarios y las preferencias de juego por género.

3. **Conclusiones y Hallazgos**:
   - El análisis ETL y EDA proporcionó información valiosa sobre la calidad de los datos, así como insights significativos sobre el contenido disponible en la plataforma y el comportamiento de los usuarios.
   - Se identificaron áreas de mejora y oportunidades para optimizar las estrategias de negocio, mejorar la experiencia del usuario y aumentar la participación.
   - Los hallazgos obtenidos a partir del análisis ETL y EDA sirvieron como base para el desarrollo de las funciones de machine learning y el sistema de recomendaciones, asegurando que estén respaldadas por datos sólidos y análisis fundamentales.

## Uso

Se muestran a continuación las funciones que se desarrollaron, junto con una descripción más detallada de cada una:

### Funciones del Proyecto:

1. **developer(desarrollador: str)**: 
   Devuelve la cantidad de ítems y el porcentaje de contenido gratuito por año, según la empresa desarrolladora.
   - **Ejemplo de retorno**:
   ```json
   {
       "Año": 2023,
       "Cantidad de Items": 50,
       "Contenido Free": "27%"
   }

2. **userdata(User_id: str)**: 
   Devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación basado en `reviews.recommend` y la cantidad de ítems.
   - **Ejemplo de retorno**:
   ```json
   {
       "Usuario X": "us213ndjss09sdf",
       "Dinero gastado": "200 USD",
       "% de recomendación": "20%",
       "cantidad de items": 5
   }

3. **UserForGenre(genero: str)**: 
   Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
   - **Ejemplo de retorno**:
   ```json
   {
    "Usuario con más horas jugadas para Género X": "us213ndjss09sdf",
    "Horas jugadas": [
        {"Año": 2013, "Horas": 203},
        {"Año": 2012, "Horas": 100},
        {"Año": 2011, "Horas": 23}
    ]
}

4. **best_developer_year(año: int)**: 
    Devuelve el top 3 de desarrolladores con juegos más recomendados por los usuarios para el año dado (considerando reseñas positivas).
    - **Ejemplo de retorno**:
    ```json
    [
    {"Puesto 1": "Desarrollador A"},
    {"Puesto 2": "Desarrollador B"},
    {"Puesto 3": "Desarrollador C"}
]

5. **developer_reviews_analysis(desarrolladora: str)**: 
    Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas categorizadas con un análisis de sentimiento positivo o negativo.
    - **Ejemplo de retorno**:
    ```json
    [

    "Valve": [Negative: 182, Positive: 278]


]

Estas funciones proporcionan funcionalidades clave para el proyecto, desde analizar el contenido según la empresa desarrolladora hasta proporcionar información detallada sobre el comportamiento del usuario y sus preferencias en el sistema de recomendaciones.

## Contacto

Cristian Andrés Riveros Cubillos - cristian1028andres@hotmail.com