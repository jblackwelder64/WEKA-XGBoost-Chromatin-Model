# WEKA-XGBoost-Chromatin-Model

This model is intended to find instances of chromatin bridges (see https://en.wikipedia.org/wiki/Chromatin_bridge for reference) in a DAPI-stained fluorescence microscopy image and then output the predicted masks as well as some statistics to indicate the frequency and density with which chromatin bridges occur.

For each image, skimage.feature.multiscale_basic_features is used to generate a 24 deep feature stack. For each location in the image, an XGBoost model uses the corresponding 24 dimensional feature vector to output the probability of that pixel belonging to a chromatin bridge. All pixels with probability higher than some threshold value (for now .65) are said to be part of a chromatin bridge. Finally, DBSCAN is used to cluster these pixels into chromatin bridge instances and remove noise.  

## Installation

Clone the repository and install all packages that are used in a conda environment.

## Usage

First, ensure that the correct conda environment is activated. Run:
```bash
conda activate cellpose-env
```

Then, use the cd command within the command line to get to the project folder, i.e. WEKA-XGBoost-Chromatin-Model.

Ensure that all ND2 files containing the images to be segmented are in a directory immediately within the project folder. Now generate the projections:
```bash
python generate_projections.py [the-nd2-files-folder-name]
```

Finally, run the model:
```bash
python find_bridges.py
```

Model outputs will be contained in a folder called 'Predictions'.

## Sample Output

Below is the model's segmentation results superimposed over the original sample image before post-processing.

<img src="Sample Predictions/Plate_ePB_v1_bulk_20220704_WellB4_ChannelDAPI,DsRed,Cy5_Seq0007 - Denoised.nd2fov_1.tif_PRED.png">

DBSCAN is used to identify chromatin bridge instances and remove noise.

<img src="Sample Predictions/Plate_ePB_v1_bulk_20220704_WellB4_ChannelDAPI,DsRed,Cy5_Seq0007 - Denoised.nd2fov_1.tif_PRED_INSTANCE_MASK.png">

Various statistics are outputted to a .csv file in order to help quantify the distribution of chromatin bridges in each image.

<img src="/Sample Statistics Output.png">



