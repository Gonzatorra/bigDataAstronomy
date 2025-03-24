from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import requests
import os

# Crear una sesión de Spark
spark = SparkSession.builder.appName("Descarga_imagenes_SDSS").getOrCreate()

# Definir las rutas de los archivos CSV
path2 = "/home/haizeagonzalez/bigData/primaryObjs_reduced.csv"

# Crear carpetas para guardar imágenes de galaxias y no galaxias
if not os.path.exists("/home/haizeagonzalez/bigData/images_galaxias"):
    os.makedirs("/home/haizeagonzalez/bigData/images_galaxias")

if not os.path.exists("/home/haizeagonzalez/bigData/images_no_galaxias"):
    os.makedirs("/home/haizeagonzalez/bigData/images_no_galaxias")

# Cargar el dataset usando PySpark
df = spark.read.csv(path2, header=True, inferSchema=True)

# Función para descargar imágenes de SDSS
def download_sdss_image(ra, dec, objID, is_galaxy):
    objID_str = str(int(objID))  # Convertir el objID a entero y luego a string
    url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.2&width=512&height=512"
    folder = "/home/haizeagonzalez/bigData/images_galaxias" if is_galaxy else "/home/haizeagonzalez/bigData/images_no_galaxias"

    try:
        # Descargar la imagen
        img_data = requests.get(url, timeout=10).content  # Añadir timeout
        with open(f"{folder}/{objID_str}.jpg", "wb") as handler:
            handler.write(img_data)
        print(f"Imagen de {objID_str} descargada en la carpeta {'galaxias' if is_galaxy else 'no_galaxias'}.")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar {objID_str}: {e}")

# Filtrar y procesar los datos usando PySpark
df.filter(col("type") == 3).select("ra", "dec", "objID").rdd.foreach(lambda row: download_sdss_image(row["ra"], row["dec"], row["objID"], True))  # Galaxias
df.filter(col("type") == 6).select("ra", "dec", "objID").rdd.foreach(lambda row: download_sdss_image(row["ra"], row["dec"], row["objID"], False))  # No galaxias

print("Descarga de imágenes completada.")
