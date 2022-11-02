# WEKA-XGBoost-Chromatin-Model

This model is intended to find instances of chromatin bridges in a fluorescence microscopy image (in particular the DAPI channel) and then output the predicted masks and some statistics to indicate the frequency and density with which chromatin bridges occur.

## Installation

Detailed instructions tbd. For now, clone the repository and install all packages that are used.

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

<img src="Predictions/Plate_ePB_v1_bulk_20220704_WellB4_ChannelDAPI,DsRed,Cy5_Seq0007 - Denoised.nd2fov_1.tif_PRED.png">
