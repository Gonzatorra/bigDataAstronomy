OBJECTIVES: Create a machine learning model for object classification and then, try and compare it with a convolutional neural network (CNN) for images. For that, I will use Spark MLlib to train and evaluate the model. Secondly, create the CNN and compare both.

HOW TO RUN IT:
You have also to change the config file (src folder). There are 3 paths:
1. DATA_PATH: Where is the csv
2. IMAGES_PATH: Where are the images (in the jupiter notebook is explained how is the schema for the images folder). The distributions of folders are made automatically with the code but you have to specify the root folder.
3. ORIGINAL_PATH: The main directory. Inside of this it has to be the DATA_PATH and the IMAGES_PATH

Example:
DATA_PATH = "/home/haizeagonzalez/bigDataProject/primaryObjs.csv"

IMAGES_PATH = "/home/haizeagonzalez/bigDataProject/images"

ORIGINAL_PATH = "/home/haizeagonzalez/bigDataProject"