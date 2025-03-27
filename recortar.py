import pandas as pd

# Leer el archivo CSV completo
csv_path = "/home/haizeagonzalez/myproject/primaryObjs_reduced.csv"  # Reemplaza con el nombre correcto

df = pd.read_csv(csv_path)

# Recortar solo las primeras 100 filas
df_recortado = df.head(100)

# Guardar el nuevo CSV
df_recortado.to_csv("/home/haizeagonzalez/myproject/recortado.csv", index=False)