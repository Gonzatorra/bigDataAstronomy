import cv2
import os
import glob

# Directorios de entrada y salida
input_dir = "/home/haizeagonzalez/bigData/imagenes"  # Carpeta con las imágenes originales
output_dir = "/home/haizeagonzalez/bigData/imagenesRecortadas"  # Carpeta para guardar imágenes recortadas

# Crear la carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

def crop_center(image_path, crop_size=64):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    cropped_img = img[center_y - crop_size//2:center_y + crop_size//2,
                      center_x - crop_size//2:center_x + crop_size//2]
    return cropped_img


# Procesar todas las imágenes
image_paths = glob.glob(os.path.join(input_dir, "*.*"))
print(image_paths)

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    cropped_img = crop_center(img_path)
    cv2.imwrite(os.path.join(output_dir, img_name), cropped_img)

print(f"Proceso de recorte completado para {len(image_paths)} imágenes.")
