To take the data you have to enter the following link:
https://skyserver.sdss.org/CasJobs/SubmitJob.aspx

and create this query:
SELECT objID, ra, dec, type,  
       petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z,  
       modelMag_u, modelMag_g, modelMag_r, modelMag_i, modelMag_z,  
       psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z,  
       (modelMag_u - modelMag_g) AS u_g,  
       (modelMag_g - modelMag_r) AS g_r,  
       (modelMag_r - modelMag_i) AS r_i,  
       (modelMag_i - modelMag_z) AS i_z,  
       fracDeV_u, fracDeV_g, fracDeV_r, fracDeV_i, fracDeV_z,  
       flags, clean  
INTO mydb.primaryObjs  
FROM dr16.PhotoPrimary  
WHERE type IN (3, 6, 1)  
AND clean = 1  
AND modelMag_r BETWEEN 14 AND 22  
AND petroRad_r > 0

You have also to change the config file. There are 3 paths:
1. DATA_PATH: Where is the csv
2. IMAGES_PATH: Where are the images (in the jupiter notebook is explained how is the schema for the images folder)
3. ORIGINAL_PATH: The main directory. Inside of this it has to be the DATA_PATH and the IMAGES_PATH

Example:
DATA_PATH = "/home/haizeagonzalez/bigDataProject/primaryObjs.csv"

IMAGES_PATH = "/home/haizeagonzalez/bigDataProject/images"

ORIGINAL_PATH = "/home/haizeagonzalez/bigDataProject"


The used libraries are:
- pyspark
- matplotlib
- pandas
- seaborn
- numpy
- scikit-learn (sklearn)
- torch
- torchvision
- random
- opencv
- scipy
