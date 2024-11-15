import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        # 다중 클래스에서 각 클래스의 precision을 계산
        precisions = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-7)
        return precisions

    def Pixel_Recall_Rate(self):
        # 다중 클래스에서 각 클래스의 recall을 계산
        recalls = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-7)
        return recalls
    
    def Pixel_F1_score(self):
        # 각 클래스별로 Precision, Recall 계산 후 F1 Score를 계산
        precisions = self.Pixel_Precision_Rate()
        recalls = self.Pixel_Recall_Rate()
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores

    def Mean_Pixel_F1_score(self):
        f1_scores = self.Pixel_F1_score()
        return np.mean(f1_scores)

    def Intersection_over_Union(self):
        # 각 클래스에 대한 IoU 계산
        ious = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        return ious

    def Mean_Intersection_over_Union(self):
        ious = self.Intersection_over_Union()
        return np.mean(ious)

    def Kappa_coefficient(self):
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)
    
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        
    def out_matrix(self):
    	confusion_matrix = self.confusion_matrix
    	return confusion_matrix
