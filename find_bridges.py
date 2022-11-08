import numpy as np
import os
from cellpose.io import imread
import xgboost as xgb
import tifffile
import matplotlib.pyplot as plt
import cellpose.models
import csv


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

def get_chromatin_pixel_count(prediction):
    # print(prediction.count(1))
    return prediction.count(1)

def get_nucleus_pixel_count(cp_model, feature_stack):
    pred2 = cp_model.eval(np.moveaxis(np.array(feature_stack), 2, 0)[0])

    nucleus_pixel_count = len([pixel_val for pixel_val in list(pred2[0].flatten()) if pixel_val>0])
    total_nuclei_count = len(np.unique(pred2[0].flatten())) - 1
    
    return nucleus_pixel_count, total_nuclei_count


def get_bridge_count(prediction, img_filename):
    """
    Using the DBSCAN clustering algorithm to 
        i. estimate the total number of Chromatin Bridges
        ii. provide a more conservative estimate for total total number of pixels in a Chromatin Bridge 
    """
    from sklearn.cluster import DBSCAN


    img = np.reshape(np.array(prediction), (2048, 2048))
    img_array = np.array(img)
    # print(np.unique(img_array))
    X = []
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            if img_array[i,j] >0:
                X.append(np.array([i,j]))        
    

    DBSCAN_cluster = DBSCAN(metric='euclidean', eps=30, min_samples=20).fit(X)
    DBSCAN_labels = DBSCAN_cluster.labels_
    
    total_bridge_count = len(np.unique(DBSCAN_labels))-1
    conservative_chromatin_pixel_count = len([label for label in DBSCAN_labels if not label==-1])

    # y_coords = [2048-X[i][0] for i in range(len(X)) if not DBSCAN_labels[i]==-1]
    # x_coords = [X[i][1] for i in range(len(X)) if not DBSCAN_labels[i]==-1]
    # plt.scatter(
    # x_coords,y_coords,s=.5,c=[label for label in DBSCAN_labels if not label==-1],
    # cmap='hsv')    
    # plt.savefig('./Predictions/'+img_filename+'_PRED_INSTANCE_MASK.png')

    pred_processed = np.zeros((len(img_array), len(img_array[0])))
    for i in range(len(X)):
        r = X[i][0]
        c = X[i][1]
        label = DBSCAN_labels[i]
        if label>0:
            pred_processed[r,c] = label

    plt.imsave('./Predictions/'+img_filename+'_PRED_INSTANCE_MASK.png', pred_processed)

    return total_bridge_count, conservative_chromatin_pixel_count





def generate_stats(feature_stack, prediction, img_filename, cp_model):
    """
    -Percentage of the entire image that is considered to be a chromosome bridge
    -Ratio of chromosome bridge pixels to nuclei pixels
    -Number of connected components
    -Ratio of # chromosome bridge connected components to # of nuclei
    """

    # cp_model = cellpose.models.CellposeModel(model_type='nuclei')
    chromatin_pixel_count = get_chromatin_pixel_count(prediction)
    nucleus_pixel_count, total_nuclei_count = get_nucleus_pixel_count(cp_model, feature_stack)

    chromatin_nucleus_pixel_ratio = chromatin_pixel_count/nucleus_pixel_count

    total_bridge_count, conservative_chromatin_pixel_count = get_bridge_count(prediction, img_filename)

    conservative_chromatin_nucleus_pixel_ratio = conservative_chromatin_pixel_count/nucleus_pixel_count
    chromatin_nucleus_instance_ratio = total_bridge_count/total_nuclei_count


    # Used https://www.codingem.com/python-write-to-csv-file/

    data = [chromatin_pixel_count, nucleus_pixel_count, chromatin_nucleus_pixel_ratio, conservative_chromatin_pixel_count, 
    conservative_chromatin_nucleus_pixel_ratio, total_bridge_count, total_nuclei_count, chromatin_nucleus_instance_ratio]

    with open('./Predictions/'+img_filename+'.csv', 'w') as file:
        fieldnames = [
            'Chromatin Pixel Count', 'Nucleus Pixel Count', 'Chromatin Nucleus Pixel Ratio', 
            'Postprocessed Chromatin Pixel Count', 'Postprocessed Chromatin Nucleus Pixel Ratio', 
            'Total Bridge Count', 'Total Nucleus Count', 'Chromatin Nucleus Instance Ratio'
            ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(zip(fieldnames, data)))
       

    


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
        img_filename = img_filenames[i]
        img = normalize_image(imgs[i])
        print('making feature stack ',i)
        print('image shape: ',img[0].shape)
        feature_stack = skimage.feature.multiscale_basic_features(img[0])
        print('finished making feature stack ',i)

        print('generating prediction',i)
        prediction = generate_prediction(feature_stack, bst)
        print('finished generating prediction',i)

        print('generating output',i)
        cp_model = cellpose.models.CellposeModel(model_type='nuclei')
        generate_stats(feature_stack, prediction, img_filename, cp_model)

        superimposed = skimage.color.label2rgb(
            label=np.reshape(np.array(prediction), (2048, 2048)), 
            image=img[0], 
            image_alpha=.65, 
            colors=[(1,1,1)])


        plt.imsave('./Predictions/'+img_filename+'_PRED.png',\
                           superimposed)
        print('finished generating output',i)

        

    

    

    













    
            
