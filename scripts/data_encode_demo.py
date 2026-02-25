import os
from os.path import join
from skimage import io
from tool.utils import structual_encoding, structual_encoding
if __name__ == '__main__':
    data_base = '/data/to/your/path/'
    stack_name = 'synapse178_labels'
    label_stack = io.imread(join(data_base, stack_name + '.tif'))

    label_save = structual_encoding(label_stack, structual_size=1, ani_scale=5)
    io.imsave(join(data_base, stack_name + '_encode.tif'), label_save)
    print('ok')
