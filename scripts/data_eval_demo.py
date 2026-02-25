import os
from os.path import join
from skimage import io
from tool.utils import structual_encoding, structual_encoding, calculate_semantics, calculate_instance
if __name__ == '__main__':
    y_pred = io.imread("/data2/share/for_gjy/SynReEM/datasets/AC3AC4/OriData/AC4Labels.tif")
    y_true = io.imread("/data2/share/for_gjy/SynReEM/datasets/AC3AC4/OriData/AC4Labels.tif")
    print('===== Semantics =====')
    calculate_semantics(y_true=y_true, y_pred=y_pred)
    print('===== Instances =====')
    calculate_instance(y_true=y_true, y_pred=y_pred)