"""
Run only once.
Generating maximum intensity projections for each field of view for each nd2 file in directory. Helpful references:
https://stackoverflow.com/questions/48178916/maximum-intensity-projection-from-image-stack
https://gist.github.com/ax3l/5781ce80b19d7df3f549
"""

import tifffile
import numpy as np
import os
import nd2
import sys

def generate_projections(directory_str):

    image_arrays = []
    filenames = []

    # directory_str = './ND2-Files-8-24-22/'
    directory = os.fsencode(directory_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.nd2'):
            image_array = nd2.imread(directory_str+'/'+filename)
            image_arrays.append(image_array)
            filenames.append(filename)

    if not os.path.exists('./Projections'):
        os.makedirs('./Projections')

    for img_ind in range(len(image_arrays)): #image_array in image_arrays:
        image_array = image_arrays[img_ind]
        filename = filenames[img_ind]
        for fov_ind in range(len(image_array)):
            fov = image_array[fov_ind]
            temp = []
            for i in range(3):
                channel = fov[:,i,:,:]
                fov_channel_max = np.max(channel, axis=0)
                temp.append(fov_channel_max)
            tifffile.imwrite('./Projections/'+filename+'fov_'+str(fov_ind+1)+'.tif',\
                            temp, photometric='minisblack')

if __name__ == '__main__':
    generate_projections(*sys.argv[1:])