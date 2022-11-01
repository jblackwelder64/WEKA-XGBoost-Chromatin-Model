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
    
    print('feature stack shape: ',feature_stack.shape)
    for i in range(2048):
        for j in range(2048):

            result = []
            for layer in feature_stack[i][j]:
                result.append(layer)     
            
            X_img.append(result)
            
    return np.array(X_img)

def normalize_image(img):
    img_copy = img.copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
    #         print(img[i][j])
    #         print(img[i][j]/65)
            img_copy[i][j] = img[i][j]/65535
    return img_copy



def generate_prediction(feature_stack, bst, threshold=.65):
    # bst = xgb.Booster()
    # bst.load_model('./Models/model3.json')
    deval = xgb.DMatrix(get_X_for_img(feature_stack))
    pred = bst.predict(data=deval)
    
    return [0 if val<threshold else 1 for val in pred]

def generate_stats(feature_stack, prediction):
    """
    -Percentage of the entire image that is considered to be a chromosome bridge
    -Ratio of chromosome bridge pixels to nuclei pixels
    -Number of connected components
    -Ratio of # chromosome bridge connected components to # of nuclei
    """
    pass


if __name__ == '__main__':

    import skimage

    # results = []
    # imgs, filenames = import_data('./Projections')
    imgs, img_filenames = import_data('./Projections/')
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
    bst.load_model('./Models/model5.json')

    print('model loaded')

    """
    The main script
    """

    
    for i in range(len(imgs)):
        # img = imgs[i]
        # img_filename = filenames[i]
        img_filename = img_filenames[i]
        img = normalize_image(imgs[i])
        print('making feature stack ',i)
        print(type(img[0]))
        print(img[0].shape)
        feature_stack = skimage.feature.multiscale_basic_features(img[0])
        print('made feature stack ',i)
        prediction = generate_prediction(feature_stack, bst)
        # results.append(prediction)

        # tifffile.imwrite('./Predictions/'+stack_filename+'_PRED.tif',\
        #                     np.reshape(np.array(prediction), (2048, 2048)), photometric='minisblack')

        plt.imsave('./Predictions/'+img_filename+'_PRED.png',\
                           np.reshape(np.array(prediction), (2048, 2048)))

        

    

    

    













    
            
