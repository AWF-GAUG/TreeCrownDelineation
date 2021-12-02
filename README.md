# Individual Tree Crown Delineation via Neural Networks

![]()

This package performs automatic delineation of individual tree crowns in remote sensing imagery. It has been tested with 30cm WordView-3 images, as well as 5cm aerial images. The contained method is ready for large scale application.

This package is under development and feedback, reported issues and contributions are very welcome!

## 1 Installation
Currently there is no pypi or conda package to install from. You therefore have to clone the package and install manually. Gdal is needed for the installation, which is easiest to install via conda.

```
# (optional) create new conda env
conda create -n <env-name>
conda activate <env-name>
conda install gdal  # install gdal upfront

cd <path where you want to keep the package> 
git clone git@github.com:AWF-GAUG/TreeCrownDelineation.git
cd ./TreeCrownDelineation

# run the package installation, which will also install the dependencies
python ./setup.py install
```

## 2 Training a model from scratch

The package is designed to work with georeferenced imagery and vector data. This means you can select your training plots in e.g. QGIS, delineate the trees within, save a shapefile containing the tree crown polygons, rasterize the polyongs into masks using the shipped scripts and run the training.

## 3 Using pre-trained models

## 4 Application / Inference

## 5 Evaluation