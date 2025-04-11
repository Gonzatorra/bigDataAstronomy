import requests
import os
import pandas as pd
import random
import config

#Get the CSV
csv_path = config.DATA_PATH
df = pd.read_csv(csv_path, dtype={"objID": str}) #We must change the objID into string to download correctly the images
df = df[['objID', 'ra', 'dec', 'type']] #Get just the necessary columns

df['objID'] = df['objID'].astype(str) #Ensure that the ID is a string
df['ra'] = df['ra'].astype(str)
df['dec'] = df['dec'].astype(str)
df['type'] = df['type'].astype(int)

#Divide the data into train and test
train_df = df.sample(frac=0.8, random_state=132)
test_df = df.drop(train_df.index) 

if not os.path.exists(config.IMAGES_PATH):
    os.makedirs("images")

for folder in ["train", "test", "validation"]:
    for obj_type in ["galaxy", "star"]:
        path = f"images/{folder}/{obj_type}"
        os.makedirs(path, exist_ok=True)

#Function to download the images
def download_sdss_image(ra, dec, objID, obj_type, folder):    
    url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=0.2&width=64&height=64"
    save_path = f"images/{folder}/{obj_type}/{objID}.jpg"

    print(f"Downloading {objID}") #Debugging print
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as handler:
                handler.write(response.content)
            print(f"Image {objID} ({obj_type}, {folder}) downloaded.")
            return True
        else:
            print(f"Failed to download {objID}. HTTP Status: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error downloading {objID}: {e}")
        return False



#Download images for training
success_count = 0
failure_count = 0

for _, row in train_df.iterrows():
    obj_type = "galaxy" if row["type"] == 3 else "star"
    if download_sdss_image(row["ra"], row["dec"], row["objID"], obj_type, "train"):
        success_count += 1
    else:
        failure_count += 1

print(f"Total downloaded: {success_count}")
print(f"Failed downloads: {failure_count}")

#Download images for testing
for _, row in test_df.iterrows():
    obj_type = "galaxy" if row["type"] == 3 else "star"
    download_sdss_image(row["ra"], row["dec"], row["objID"], obj_type, "test")

print("All images downloaded.")