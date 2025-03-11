import pandas as pd
import requests
import os

# Crear carpeta para guardar imágenes
if not os.path.exists("images"):
    os.makedirs("images")

# Cargar el dataset descargado de SDSS
df = pd.read_csv("sdss_galaxies.csv")  # Asegúrate de que el archivo está en la misma carpeta del script

# Función para descargar imágenes de SDSS
def download_sdss_image(ra, dec, objID):
    url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.2&width=512&height=512"
    try:
        img_data = requests.get(url).content
        with open(f"images/{objID}.jpg", "wb") as handler:
            handler.write(img_data)
        print(f"Imagen de {objID} descargada.")
    except Exception as e:
        print(f"Error al descargar {objID}: {e}")

# Descargar imágenes para todas las galaxias (puedes ajustar el número)
for index, row in df.iterrows():
    download_sdss_image(row["ra"], row["dec"], row["objID"])

print("Descarga de imágenes completada.")
