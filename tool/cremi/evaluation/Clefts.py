import numpy as np
from scipy import ndimage

class Clefts:

    def __init__(self, test, truth):

        test_clefts = test
        truth_clefts = truth
        # 取出真值的背景，赋值为True
        # 0xfffffffffffffffe在np.int64中是-1
        self.truth_clefts_invalid = truth_clefts.data[()] == 0xfffffffffffffffe
        # numpy.logical_or(x1, x2)
        # 返回X1和X2或逻辑后的布尔值。
        self.test_clefts_mask = np.logical_or(test_clefts.data[()] == 0xffffffffffffffff, self.truth_clefts_invalid)
        self.truth_clefts_mask = np.logical_or(truth_clefts.data[()] == 0xffffffffffffffff, self.truth_clefts_invalid)
        # ndimage.distance_transform_edt:前景到背景的距离
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.distance_transform_edt.html
        # sampling各个维度的标度尺，这里的参数在Volume.py中，resolution=(1.0, 1.0, 1.0)
        self.test_clefts_edt = ndimage.distance_transform_edt(self.test_clefts_mask, sampling=test_clefts.resolution)
        self.truth_clefts_edt = ndimage.distance_transform_edt(self.truth_clefts_mask, sampling=truth_clefts.resolution)

    def count_false_positives(self, threshold=200):
        # 按位取反，使得背景为False，实例为True
        mask1 = np.invert(self.test_clefts_mask)
        mask2 = self.truth_clefts_edt > threshold
        false_positives = self.truth_clefts_edt[np.logical_and(mask1, mask2)]
        return false_positives.size

    def count_false_negatives(self, threshold=200):
        # 对真值取反，原本1背景0突触，后0背景1突触
        mask1 = np.invert(self.truth_clefts_mask)
        # 对预测掩膜进行增长，1背景0突触，0的区域增加
        mask2 = self.test_clefts_edt > threshold
        # FN：即真值为0(突触)，预测为1(背景)
        false_negatives = self.test_clefts_edt[np.logical_and(mask1, mask2)]
        # 这里其实只需要np.sum(np.logical_and(mask1, mask2))即可
        return false_negatives.size

    def count_true_positives(self, threshold=200):
        # i.e., any detection voxel inside a grown ground truth region is counted as a true positive.
        # 先增长，再转换
        mask_test = np.invert(self.test_clefts_mask)
        mask_truth = np.invert(self.truth_clefts_edt > threshold)
        true_positives = np.sum(np.logical_and(mask_test, mask_truth))
        # # 真值与预测值原本重合的
        # mask_test_1 = np.invert(self.test_clefts_mask)
        # mask_truth_1 = np.invert(self.truth_clefts_mask)
        # true_positives_1 = np.sum(np.logical_and(mask_test_1, mask_truth_1))
        # # 预测增长的与真值重合的
        # mask_test_2 = np.invert(self.test_clefts_edt > threshold)
        # mask_truth_2 = np.invert(self.truth_clefts_mask)
        # true_positives_2 = np.sum(np.logical_and(mask_test_2, mask_truth_2))
        # # 真值增长的与预测重合的
        # mask_test_3 = np.invert(self.test_clefts_mask)
        # mask_truth_3 = np.invert(self.truth_clefts_edt > threshold)
        # true_positives_3 = np.sum(np.logical_and(mask_test_3, mask_truth_3))

        return true_positives

    def acc_false_positives(self):

        mask = np.invert(self.test_clefts_mask)
        false_positives = self.truth_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_positives),
            'std': np.std(false_positives),
            'max': np.amax(false_positives),
            'count': false_positives.size,
            'median': np.median(false_positives)}
        return stats

    def acc_false_negatives(self):

        mask = np.invert(self.truth_clefts_mask)
        false_negatives = self.test_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_negatives),
            'std': np.std(false_negatives),
            'max': np.amax(false_negatives),
            'count': false_negatives.size,
            'median': np.median(false_negatives)}
        return stats

