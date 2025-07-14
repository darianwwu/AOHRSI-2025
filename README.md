# AOHRSI-2025

Welcome to our repository for road segmentation. We are comparing machine learning approaches with traditional feauture extractions.
This repository documents the workflow and our code.


# Data
All data for training and validation was downloaded from the GEOportal.NRW (Link: https://www.geoportal.nrw/).
The data comes from different cities in North-Rhine-Westphalia and from the dataset "	InVeKoS Digitale Orthophotos (2-fache Kompression) – Paketierung: Einzelkacheln".
The data was collected in early 2025 and it has a spatial resolution of 0,2 x 0,2 m.


## Machine Learning Approach
# Requirements:
Install QGis (Link: https://plugins.qgis.org/plugins/deepness/)

Install QuickOSM plugin for QGis:
1. Open QGis
2. Go to Plugins > Manage and Install Plugins > All
3. Search for "QuickOSM"
4. Click on "Install Plugin"
5. Restart QGis

Have an environment to run python Code (e.g. Visual Studio Code, Link: https://code.visualstudio.com/)

Additional (for validation and comparison)
Install QGis Deepness Plugin (Link: https://plugins.qgis.org/plugins/deepness/)
Download Deepness Road Segmentation Model (Link: https://qgis-plugin-deepness.readthedocs.io/en/latest/main/main_model_zoo.html)
1. Open QGis
2. Go to Plugins > Manage and Install Plugins > Install from ZIP
3. Select the downloaded deepness file and install it

If the plugin installation fails to load packages, do the following:
1. Close QGis
2. Go to the folder: C:\Users\<your_username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\deepness\python3.12\
3. Delete this folders if they exist: onnxruntime, onnxruntime<version>.dist-info

Install required Python packages manually
1. Reopen QGis
2. Open Plugins > Python Console
3. Install the required packages by running this code:
import pip
pip.main(['install', 'onnxruntime'])
pip.main(['install', 'opencv-contrib-python'])
pip.main(['install', 'torch'])
pip.main(['install', 'Pillow'])

# Preparing the data
What you have: An orthophoto with a spatial resolution of 0,2 x 0,2 m.

# Step 1 - Get the OSM street data
1. Open QGis
2. Add the orthophoto to the project
3. Go to Vector > QuickOSM > QuickOSM > Quick query
4. Fill in the following values: Key - highway, Layer extent, [Your orthophoto] (See image below)
5. Click "Run query"

# Step 2 - Get the data in the right reference system
1. Right click on the highway line feature
2. Click on Export > Save Features as
3. Give it a name and for CRS pick: "Project CRS: EPSG:25832 - ETRS89 / UTM zone 32N
4. Click on OK

# Step 3 - Buffer the lines
1. Go to Vector > Geoprocessing Tools > Buffer
2. Select your reprojected line layer as the input layer
3. Select "Distance" 4 Meters (<- not degrees)
4. Click on Run

# Step 4 - Convert Vector to Raster
1. Go to Raster > Conversion > Rasterize (Vector to Raster)
2. Select your buffered feature
3. Input 1 for A fixed value to burn
4. For "Output Raster Size Units" pick "Georeferenced Units"
5. Put in 0,2 for Height and Width
6. Click on Run

# Step 5 - Cut to the right size
1. Go to Raster > Extraction > Clip Raster by Extent
2. Select your rasterized feature as the input layer
3. For clipping extent select Calculate from Layer > your orthophoto
4. Click on Run

# Step 6 - Preparing for testing/ training or validation
1. Open your python IDE
2. Copy the orthophoto and the clipped extent to the inputs directory

# Step 7.0 - Running a prediction on an orthophoto
1. Open predict.py
2. Adjust the value for INPUT_TIF to match your orthophoto path
3. Save your changes
4. Run the script by typing python predict.py in the console
5. The models prediction is stored in the outputs/ directory

# Step 7.1 - Evaluating the model
1. In prepare.py: Adjust the values for ortho_tif_path and osm_tif_path
2. Run prepare.py by typing this command: python prepare.py
3. The training data tiles are saved in the training_data/ directory
4. Move the tiles/ folder from the training_data/ directory to the validation/ directory
5. Run evaluate.py by typing this command: python evaluate.py
6. The evaluation metrcs will be printed out in the console

# Step 7.2 - OPTIONAL: Compare with Deepness model
1. Run evaluateonnx.py by typing this command: python evaluateonnx.py
2. The evaluation metrcs will be printed out in the console

# Step 7.3 - Finetune the model by training it again
1. Get a new orthophoto and OSM data (TIFs) as described in the steps above
2. Do step 6
3. In prepare.py: Adjust the values for ortho_tif_path and osm_tif_path
4. Run prepare.py by typing this command: python prepare.py
5. The training data tiles are saved in the training_data directory
6. Run train.py by typing this command: python train.py
7. After some time, the finetuned model is saved in models/roadsegmentation_model_finetuned.pth

