from math import exp
import numpy as np
import cv2
import os
import imgproc
from shapely.geometry import LineString, Polygon
import config


def showmat(name, mat):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, mat)
    cv2.waitKey(0)


class GaussianTransformer(object):

    def __init__(self, imgSize=512):
        # distanceRatio = 3.34
        self.imgSize = imgSize
        self.standardGaussianHeat = self._gen_gaussian_heatmap(imgSize)
        # print("thresh: ",region_threshold,affinity_threshold)
        # _, binary = cv2.threshold(self.standardGaussianHeat, region_threshold * 255, 255, 0)
        # np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        start_position_region = config.start_position_region
        start_position_affinity = config.start_position_affinity
        size_of_heatmap_region = config.size_of_heatmap_region
        size_of_heatmap_affinity = config.size_of_heatmap_affinity
        x_r, y_r, w_r, h_r = int(start_position_region * imgSize), int(start_position_region * imgSize), int(size_of_heatmap_region * imgSize), int(
            size_of_heatmap_region * imgSize)  # cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x_r, y_r], [x_r + w_r, y_r], [x_r + w_r, y_r + h_r], [x_r, y_r + h_r]], dtype=np.int32)
        # print("regionbox", self.regionbox)
        # img_reg=self.standardGaussianHeat[y:y+h,x:x+w]
        # x,y,w,h=cv2.boundingRect(np_contours)
        # img_reg_1=self.standardGaussianHeat[y:y+h,x:x+w]
        # showmat("reg",img_reg)
        # showmat("reg1",img_reg_1)
        # showmat("src",self.standardGaussianHeat)
        # _, binary = cv2.threshold(self.standardGaussianHeat, affinity_threshold * 255, 255, 0)
        # np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        # x, y, w, h = int(0.15 * imgSize), int(0.15 * imgSize), int(0.7 * imgSize), int(0.7 * imgSize)  # cv2.boundingRect(np_contours)
        # img_aff = self.standardGaussianHeat[y:y + h, x:x + w]
        # x, y, w, h = cv2.boundingRect(np_contours)
        # img_aff_1 = self.standardGaussianHeat[y:y + h, x:x + w]
        # showmat("aff", img_aff)
        # showmat("aff1", img_aff_1)
        # showmat("src_aff", self.standardGaussianHeat)
        x_a, y_a, w_a, h_a = int(start_position_affinity * imgSize), int(start_position_affinity * imgSize), int(size_of_heatmap_affinity * imgSize), int(
            size_of_heatmap_affinity * imgSize)  # cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x_a, y_a], [x_a + w_a, y_a], [x_a + w_a, y_a + h_a], [x_a, y_a + h_a]], dtype=np.int32)
        # print("affinitybox", self.affinitybox)
        self.oribox = np.array([[0, 0, 1], [imgSize - 1, 0, 1], [imgSize - 1, imgSize - 1, 1], [0, imgSize - 1, 1]],
                               dtype=np.int32)

    def _gen_gaussian_heatmap(self, imgSize):
        def scaled_gaussian(x):
                return np.exp(-(1 / 2) * (x ** 2))  # not pdf

        # heat = np.zeros((imgSize, imgSize), np.uint8)
        # for i in range(imgSize):
        #     for j in range(imgSize):
        #         distanceFromCenter = np.linalg.norm(np.array([i - imgSize / 2, j - imgSize / 2]))
        #         distanceFromCenter = distanceRatio * distanceFromCenter / (imgSize / 2)
        #         scaledGaussianProb = scaledGaussian(distanceFromCenter)
        #         heat[i, j] = np.clip(scaledGaussianProb * 255, 0, 255)
        # # print("max value: ",np.max(heat))
        x, y = np.meshgrid(np.linspace(-2.5, 2.5, imgSize),
                           np.linspace(-2.5, 2.5, imgSize))
        distance_from_center = \
            np.linalg.norm(np.stack([x, y], axis=0), axis=0, keepdims=False)
        scaled_gaussian_prob = scaled_gaussian(distance_from_center)
        heat = \
            np.clip(np.round(scaled_gaussian_prob * 255),
                    0,
                    255).astype(np.uint8)
        return heat

    def _test(self):
        sigma = 10
        spread = 3
        extent = int(spread * sigma)
        center = spread * sigma / 2
        gaussian_heatmap = np.zeros([extent, extent], dtype=np.float32)

        for i_ in range(extent):
            for j_ in range(extent):
                gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))

        gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        threshhold_guassian = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'test_guassian.jpg'), threshhold_guassian)

    def order_points(self, box, image_shape):
        blank_img = np.ones((image_shape), dtype=np.uint8) * 255
        contour = np.array(box, dtype=np.int32).reshape(4, 2)
        cv2.polylines(blank_img, [contour], True, 0)
        cnts = np.where(blank_img == 0)
        cnts = np.array([[x, y] for (y, x) in zip(cnts[0], cnts[1])])
        # rect_box = cv2.minAreaRect(cnts)
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if len(cnts) == 0:
            print("error in order_points function: ", box)
            return box
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(cnts[:, 0]), max(cnts[:, 0])
            t, b = min(cnts[:, 1]), max(cnts[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        return box

    def add_region_character(self, image, target_bbox, regionbox=None):
        image_shape = image.shape[:2]
        target_bbox = self.order_points(target_bbox, image_shape)
        # print(target_bbox)
        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True
        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        transformed = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
        # showmat("trans",np.uint8(transformed))
        image = np.where(transformed > image, transformed, image)

        # showmat("img",image)
        return image

    def add_region_character1(self, image, target_bbox, regionbox=None):

        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        oribox = np.array(
            [[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
            dtype=np.float32)
        test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
        real_target_box = cv2.perspectiveTransform(oribox, M)[0]
        # print("test\ntarget_bbox", target_bbox, "\ntest1", test1, "\nreal_target_box", real_target_box)
        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
        # warped = np.array(warped, np.uint8)
        # image = np.where(warped > image, warped, image)
        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
            # if False:
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)
        # _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
        # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
        # warped = np.array(warped, np.uint8)
        #
        # # if affi:
        # # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
        # # cv2.imshow("1123", warped)
        # # cv2.waitKey()
        # image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
        #                                        image[ymin:ymax, xmin:xmax])
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image
            # if affi:
            # print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # cv2.imshow("1123", warped)
            # cv2.waitKey()
            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
        return image

    def add_affinity_character(self, image, target_bbox):
        return self.add_region_character(image, target_bbox, self.affinitybox)

    def add_affinity(self, image, bbox_1, bbox_2,dir='h'):
        # center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        # tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        # bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        # tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        # br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)
        diagonal11 = LineString([bbox_1[0], bbox_1[2]])
        diagonal12 = LineString([bbox_1[1], bbox_1[3]])
        center_1 = diagonal11.intersection(diagonal12).coords[:][0]
        diagonal21 = LineString([bbox_2[0], bbox_2[2]])
        diagonal22 = LineString([bbox_2[1], bbox_2[3]])
        center_2 = diagonal21.intersection(diagonal22).coords[:][0]
        if dir=='h':
            tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
            bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
            tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
            br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)
        else:
            tl = np.mean([bbox_1[0], bbox_1[3], center_1], axis=0)
            bl = np.mean([bbox_1[1], bbox_1[2], center_1], axis=0)
            tr = np.mean([bbox_2[0], bbox_2[3], center_2], axis=0)
            br = np.mean([bbox_2[1], bbox_2[2], center_2], axis=0)


        affinity = np.array([tl, tr, br, bl])

        return self.add_affinity_character(image, affinity.copy()), np.expand_dims(affinity, axis=0)

    def generate_region(self, image_size, bboxes):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i].copy())
            for j in range(bboxes[i].shape[0]):
                target = self.add_region_character(target, character_bbox[j])

        return target

    def generate_affinity(self, image_size, bboxes, words,image):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            # print(bboxes[i].reshape(-1, 2))
            rect = cv2.minAreaRect(bboxes[i].reshape(-1,2).astype(np.float32))
            boxPoints=cv2.boxPoints(rect)
            xmin=min(boxPoints[:,0])
            xmax=max(boxPoints[:,0])
            ymin = min(boxPoints[:, 1])
            ymax = max(boxPoints[:, 1])
            width_box=xmax-xmin
            height_box=ymax-ymin
            # cv2.drawContours(image,[np.int32(bboxes[i].reshape(-1,2))],0,(0,0,255),2)
            # print("ratio: ",width_box/height_box)
            # showmat("img",image)
            if width_box/height_box<0.5 and len(character_bbox)>2:
                rnd_idx=np.random.randint(0,10000)
                cv2.drawContours(image, [np.int32(bboxes[i].reshape(-1,2))], 0, (0, 0, 255), 2)
                cv2.imwrite("debug/"+str(rnd_idx)+".png",image)
                dir='v'
                print("vertical")
            else:
                dir='h'

            total_letters = 0
            for char_num in range(character_bbox.shape[0] - 1):
                target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                     character_bbox[total_letters + 1],dir=dir)
                affinities.append(affinity)
                total_letters += 1
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities

    def saveGaussianHeat(self):
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        cv2.polylines(warped_color, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standardGaussianHeat1 = self.standardGaussianHeat.copy()
        standardGaussianHeat1[standardGaussianHeat1 > 0] = 255
        threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1, cv2.COLORMAP_JET)
        cv2.polylines(threshhold_guassian, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)


if __name__ == '__main__':
    gaussian = GaussianTransformer(1024)
    gaussian.saveGaussianHeat()
    bbox = np.array([[[1, 200], [510, 200], [510, 510], [1, 510]]])
    print(bbox.shape)
    bbox = bbox.transpose((2, 1, 0))
    print(bbox.shape)
#


# coding=utf-8
# coding=utf-8
