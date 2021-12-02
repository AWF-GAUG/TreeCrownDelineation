#! /usr/bin/env python3
import setuptools
from setuptools import setup, find_packages

NAME = 'treecrowndelineation'
VERSION = '1.0.0'
DESCRIPTION = 'Tree crown delineation using U-Nets and watershed transform.'
URL = 'https://github.com/AWF-GAUG/TreeCrownDelineation'
AUTHOR = 'Max Freudenberg'
LICENCE = 'GPL2.0'
LONG_DESCRIPTION = """"""

setup(name=NAME,
      version=VERSION,
      python_requires='>3.5',
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL,
      author=AUTHOR,
      license=LICENCE,
      packages=find_packages(),
      #include_package_data=True,
      install_requires=["torch",
                        "pytorch_lightning",
                        "numpy",
                        "segmentation_models_pytorch",
                        "albumentations",
                        "xarray",
                        "rasterio",
                        "rioxarray",
                        "shapely",
                        "gdal",
                        "scipy",
                        "scikit-image",
                        "hyperopt",
                        "fiona",
                        "psutil"],
      zip_safe=False)
