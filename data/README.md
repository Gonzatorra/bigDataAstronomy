DOWNLOAD DATA:
1. To take the data you have to enter the following link:
   https://skyserver.sdss.org/CasJobs/

2. Create a user.

3. 
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

