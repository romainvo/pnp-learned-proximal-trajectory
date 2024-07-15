# Walnut-CBCT

*A cone-beam X-ray computed tomography data collection designed for machine learning, 2019, Der Sarkissian et al*.

## Download

The dataset can be download here : https://zenodo.org/records/2686726. This dataset is composed of 42 .zip files. Each .zip contains .tiff files for the projections as well as already computed reconstructions, it also contains the trajectory of the source, detector and object during the acquisition.

## Setup

Once all the .zip files have been downloaded and placed in `<data>` folder, it can be processed using the `prepare_data.py` by replacing the variable `input_dir` with your `<data>` location. This script will assemble the .tiff files into .raw files easier to manipulate.

In the `<data>` dir you can then run the `create_splits.py` script, it will create the train/val/test splits as well as a `dataset_50p.csv` files to store useful informations on the geometry, and sample easily from the dataset during the experiment. The data is split into 30 train, 4 val, 8 test.

To prepare the data you will need the following libraries:
```
numpy
astra
imageio
matplotlib
pandas
zipfile
shutil
import tifffile
```

The `astra-toolbox` library can be installed by following the instructions here https://github.com/astra-toolbox/astra-toolbox.