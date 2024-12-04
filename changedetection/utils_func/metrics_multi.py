import csv
import numpy as np
from skimage.measure import label, regionprops

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
        
    def cal_tp(self, gt_label, pred_label, path, image_name, threshold=0.3):
        tp1, tp2 = 0, 0
        iou_test = path
        image_name = image_name
        
        gt_label = np.squeeze(gt_label)
        pred_label = np.squeeze(pred_label)
        # 정답 데이터에서 객체 추출
        gt_labeled = label(gt_label > 0, connectivity=1)
        gt_regions = regionprops(gt_labeled)
        total1 = int(gt_labeled.max())

        # 예측 데이터에서 객체 추출
        pred_labeled = label(pred_label > 0, connectivity=1)
        pred_regions = regionprops(pred_labeled)
        total2 = int(pred_labeled.max())
        
        for i in range(len(gt_regions)):
            gt_region = gt_regions[i]
            # 정답 객체의 클래스 및 픽셀 좌표
            gt_coords = set(map(tuple, gt_region.coords))
            gt_class = gt_label[gt_region.coords[0][0], gt_region.coords[0][1]]
            
            for j in range(len(pred_regions)):
                pred_region = pred_regions[j]
                # 예측 객체의 클래스 및 픽셀 좌표
                pred_coords = set(map(tuple, pred_region.coords))
                pred_class = pred_label[pred_region.coords[0][0], pred_region.coords[0][1]]

                # 클래스가 동일하지 않으면 비교하지 않음
                if gt_class != pred_class:
                    continue

                # 두 객체 간 중첩된 픽셀 계산
                intersection = gt_coords & pred_coords
                intersection_size = len(intersection)

                # 각 객체의 크기 계산
                gt_size = len(gt_coords)
                pred_size = len(pred_coords)

                # 중첩 비율 계산
                gt_overlap = intersection_size / gt_size
                pred_overlap = intersection_size / pred_size
                
                with open(iou_test, mode='a') as f:
                    writer = csv.writer(f)
                    writer.writerow([image_name, i, j, gt_class, f'{gt_overlap*100:.2f}%'])

                # 클래스가 동일하고 중첩 비율이 threshold 이상이면 True 반환
                if gt_overlap >= threshold:
                    tp1 += 1
                elif pred_overlap >= threshold:
                    tp2 += 1
                    
        return tp1, tp2, total1, total2, 
        
    def out_matrix(self):
    	confusion_matrix = self.confusion_matrix
    	return confusion_matrix
