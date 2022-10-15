import numpy as np
import os
from cellpose.io import imread
import xgboost as xgb
import tifffile
import matplotlib.pyplot as plt


def import_data(proj_directory_str):

    X = []
    filenames = []
    
    directory = os.fsencode(proj_directory_str)
    for file in [file for file in sorted(os.listdir(directory)) if os.fsdecode(file)[-4:] == '.tif']:
        filename = os.fsdecode(file)
        filenames.append(filename)

        image_array = imread(proj_directory_str+filename)
        

        X.append(image_array)
    
    return X, filenames


def get_X_for_img(feature_stack):
    X_img = []
    
    for i in range(2048):
        for j in range(2048):

            result = []
            for layer in feature_stack:
                result.append(layer[i,j])     
            
            X_img.append(result)
            
    return np.array(X_img)


def generate_prediction(feature_stack, bst, threshold=.65):
    # bst = xgb.Booster()
    # bst.load_model('./Models/model3.json')
    deval = xgb.DMatrix(get_X_for_img(feature_stack))
    pred = bst.predict(data=deval)
    
    return [0 if val<threshold else 1 for val in pred]

def generate_stats(prediction):
    pass


if __name__ == '__main__':
    # results = []
    # imgs, filenames = import_data('./Projections')
    img_stacks, stack_filenames = import_data('./Feature Stacks/')
    """
    for each image:
        generate predictions
        do nuclei segmentation with cellpose
        analyze predictions (i.e., compare chromatin density to cellpose density)
    Make sure to generate feature stacks via the bash script (which I still have to implement) or yourself in FIJI first.
        If doing yourself, place them in ./Feature Stacks and name them as *image-projection-filename-without-.tif* + _STACK.tif 
    """

    """
    Briefly checking that feature stacks were generated correctly
    """
    # if len(imgs) != len(img_stacks):
    #     raise ValueError('Lengths of images array and image feature stacks not equal.') 
    # for i in range(len(filenames)):
    #     img_filename = filenames[i]
    #     stack_filename = stack_filenames[i]
    #     if img_filename[0:-4] != stack_filename[0:-10]:
    #         raise ValueError('Image filename: '+img_filename, +
    #         ' does not match feature stack filename: '+stack_filename)

    """
    Importing feature stacks and generating/write predictions
    """

    if not os.path.exists('./Predictions'):
        os.makedirs('./Predictions')

    """
    Importing the model parameters
    """

    bst = xgb.Booster()
    bst.load_model('./Models/model3.json')

    
    for i in range(len(img_stacks)):
        # img = imgs[i]
        # img_filename = filenames[i]
        stack_filename = stack_filenames[i]
        img_stack = img_stacks[i]
        prediction = generate_prediction(img_stack, bst)
        # results.append(prediction)

        # tifffile.imwrite('./Predictions/'+stack_filename+'_PRED.tif',\
        #                     np.reshape(np.array(prediction), (2048, 2048)), photometric='minisblack')

        plt.imsave('./Predictions/'+stack_filename+'_PRED.png',\
                           np.reshape(np.array(prediction), (2048, 2048)))

        

    

    

    













    
            
