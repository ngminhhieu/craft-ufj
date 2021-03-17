###for icdar2015####


import torch
import torch.utils.data as data
import scipy.io as scio
from gaussian import GaussianTransformer
from watershed import watershed
import re
import itertools
from file_utils import *
from mep import mep
import random
from PIL import Image
import torchvision.transforms as transforms
import craft_utils
import Polygon as plg
import time
import glob
from pathlib import Path
import config
from craft import CRAFT
from torchutil import copyStateDict

def showmat(name, mat):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, mat)
    cv2.waitKey(0)


def check_intersec(y1, y4, yj1, yj4, intersec):
    if yj4 - yj1 < y4 - y1:
        h = yj4 - y1
    else:
        h = y4 - y1
    if intersec / h > 0.3:
        return True
    return False


def color_jitter_image():
    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.2)
    transform = transforms.ColorJitter.get_params(
        color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
        color_jitter.hue)
    return transform


def random_scale(img, bboxes, target_size=768, max_scale=3, min_scale=0.3, rnd_scale=0.5):
    if random.random() < rnd_scale:
        h, w = img.shape[0:2]
        if h > target_size or w > target_size:
            scale = random.uniform(min_scale, max_scale / 2)
        else:
            scale = random.uniform(min_scale * 2, max_scale)
        bboxes *= scale
        # print("scale: ", scale, h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def padding_image(images, target_h, target_w):
    padded_imgs = []
    actual_h, actual_w = images[0].shape[:2]
    target_h = max(target_h, actual_h)
    target_w = max(target_w, actual_w)
    pad_h = target_h - actual_h
    pad_w = target_w - actual_w
    rnd_pad_top, rnd_pad_bot, rnd_pad_right, rnd_pad_left = 0, 0, 0, 0
    if pad_h > 0:
        rnd_pad_top = random.randint(0, pad_h)
        rnd_pad_bot = pad_h - rnd_pad_top
    if pad_w > 0:
        rnd_pad_left = random.randint(0, pad_w)
        rnd_pad_right = pad_w - rnd_pad_left
    for idx, image in enumerate(images):
        mean = int(np.mean(image))
        input_dimension = len(image.shape)
        target_shape = (target_h, target_w) if input_dimension == 2 else (target_h, target_w, 3)
        big_img = np.ones(target_shape, dtype=np.uint8) * mean if idx != (len(images) - 1) else np.ones(target_shape,
                                                                                                        dtype=np.float32)
        big_img[rnd_pad_top:target_h - rnd_pad_bot, rnd_pad_right:target_w - rnd_pad_left] = image
        padded_imgs.append(big_img)
    return padded_imgs


def random_crop(imgs, img_size, scale=1.5):
    target_h, target_w = img_size
    padded_h, padded_w = int(target_h * scale), int(target_w * scale)
    imgs = padding_image(imgs, padded_h, padded_w)
    x = random.randint(0, imgs[0].shape[1] - target_w)
    y = random.randint(0, imgs[0].shape[0] - target_h)
    for idx, img in enumerate(imgs):
        img = img[y:y + target_h, x:x + target_w]
        imgs[idx] = img

    return imgs


def random_horizontal_flip(imgs, rnd_flip=0.4):
    if random.random() < rnd_flip:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs, rnd_rotate=0.3):
    max_angle = 10
    if random.random() < rnd_rotate:
        angle = random.uniform(-max_angle, max_angle)
        # print("rnd angle: ",angle)
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = img_rotation
    return imgs


class Data_LinkRefiner(data.Dataset):
    def __init__(self, folder, target_size=768):
        self.target_size = target_size
        self.img_folder = os.path.join(folder, 'imgs')
        self.gt_folder = os.path.join(folder, 'gt')
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)

    def load_gt(self, gt_fn):
        with open(gt_fn, "r") as gt_:
            lines = gt_.readlines()
        list_bboxes = []
        for line in lines:
            line = line.replace("\n", "").split(",")
            bbox = line[:-2]
            txt = line[9]
            if txt == "###":
                continue
            bbox = np.array(bbox, np.int32).reshape((4, 2))
            list_bboxes.append(bbox)
        list_bboxes = sorted(list_bboxes, key=lambda kv: kv[0][1])
        return list_bboxes

    def __getitem__(self, index):
        image, mask, list_bboxes = self.load_image_gt(index)
        list_bboxes = np.expand_dims(list_bboxes, axis=0)
        random_transforms = [image, mask]
        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size))
        random_transforms = random_horizontal_flip(random_transforms)
        random_transforms = random_rotate(random_transforms)
        image, mask = random_transforms
        mask = self.resizeGt(mask)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))

        # showmat("img-",image)
        # showmat("mask-",mask*255)
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()
        return image, mask

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def load_image_gt(self, index):
        imagename = self.images_path[index]
        # print(imagename)
        gt_path = os.path.join(self.gt_folder, "gt_%s.txt" % os.path.splitext(imagename)[0])
        list_bboxes = self.load_gt(gt_path)
        list_bboxes = np.float32(list_bboxes)
        image_path = os.path.join(self.img_folder, imagename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = random_scale(image, list_bboxes, self.target_size)
        mask = np.zeros((image.shape[0], image.shape[1]))
        checked = []
        append_bboxes = []
        center_lines = []
        for i in range(len(list_bboxes)):
            if i in checked:
                continue
            tmp_top = []
            tmp_bot = []
            checked.append(i)
            x1, y1, x2, y2, x3, y3, x4, y4 = list_bboxes[i].reshape(8).tolist()
            tmp_top.extend((x1, y1, x2, y2))
            tmp_bot.extend((x3, y3, x4, y4))
            selected = []
            selected.append(i)
            for j in range(i + 1, len(list_bboxes)):
                merge = False
                xj1, yj1, xj2, yj2, xj3, yj3, xj4, yj4 = list_bboxes[j].reshape(8).tolist()
                if (y1 <= yj1 < y4 and y1 < yj4 <= y4) or (yj1 <= y1 <= yj4 and yj1 <= y1 <= yj4):
                    merge = True
                elif yj1 < y1 and y1 < yj4 < y4:
                    intersec = yj4 - y1
                    merge = check_intersec(y2, y4, yj1, yj4, intersec)
                elif y1 < yj1 <= y4 and y4 < yj4:
                    intersec = y4 - yj1
                    merge = check_intersec(y2, y4, yj1, yj4, intersec)
                else:
                    merge = False
                if merge:
                    tmp_top.extend((xj1, yj1, xj2, yj2))
                    tmp_bot.extend((xj3, yj3, xj4, yj4))
                    checked.append(j)
                    selected.append(j)
                    y1 = min(y1, yj1)
                    y4 = max(y4, yj4)

            bbox = []
            for s in selected:
                bbox.append(list_bboxes[s])
            bbox = sorted(bbox, key=lambda kv: kv[0][0])
            tmp = []
            for box in bbox:
                x1, y1, x2, y2, x3, y3, x4, y4 = box.reshape(8).tolist()
                tmp.append((x1, y1, x2, y2))
            for box in reversed(bbox):
                x1, y1, x2, y2, x3, y3, x4, y4 = box.reshape(8).tolist()
                tmp.append((x3, y3, x4, y4))
            bbox = np.array(tmp).reshape((-1, 2))
            append_bboxes.append(bbox)

        for i in range(len(append_bboxes)):
            points = append_bboxes[i]
            num_points = len(points)
            tmp_top = []
            tmp_bot = []
            for j in range(num_points // 2):
                start = points[j]
                end = points[num_points - j - 1]
                sy = start[1]
                ey = end[1]
                w = 0.1 * (ey - sy)
                mid = (sy + ey) / 2
                s_mid = mid - w
                e_mid = mid + w
                tmp_top.append((start[0], s_mid,))
                tmp_bot.append((end[0], e_mid))
            tmp_bot.reverse()
            bboxes = tmp_top + tmp_bot
            bboxes = np.array(bboxes, np.int32).reshape((-1, 2))
            center_lines.append(bboxes)

        # print(tmp_bot)

        for i, bbox in enumerate(append_bboxes):
            # cv2.fillPoly(image, [np.int32(bbox)], (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            # cv2.fillPoly(image, [center_lines[i]], (0, 0, 255))
            cv2.fillPoly(mask, [center_lines[i]], (1))
        # cv2.imwrite("debug/"+imagename,mask*255)
        # showmat("img",image)
        for i, box in enumerate(list_bboxes):
            box = box.reshape(1, 4, 2)
            list_bboxes[i] = box
        return image, mask, list_bboxes


class craft_base_dataset(data.Dataset):
    def __init__(self, target_size=768, viz=False, debug=False):
        self.target_size = target_size
        self.viz = viz
        self.debug = debug
        gaussian_heatmap_size = config.gaussian_heatmap_size
        self.gaussianTransformer = GaussianTransformer(imgSize=gaussian_heatmap_size)

    def load_image_gt_and_confidencemask(self, index):
        return None, None, None, None, None

    def crop_image_by_bbox1(self, image, box):
        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        if h > w * 1.5:
            width = h
            height = w
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
        else:
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M

    def crop_image_by_bbox(self, image, box):
        (tl, tr, br, bl) = box
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        if maxHeight > maxWidth * 1.5:
            width = maxHeight
            height = maxWidth
            dst = np.array([
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
                [0, 0]], dtype="float32")
        else:
            width = maxWidth
            height = maxHeight
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(box, dst)
        maxHeight = height
        maxWidth = width
        warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warp, M

    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

    def inference_pursedo_bboxes(self, net, image, word_bbox, word, viz=False, idx=-1, height_of_box=64.0,
                                 expand_small_box=5):

        word_image, MM = self.crop_image_by_bbox(image, word_bbox)
        # print(word_bbox)
        # showmat("img", word_image)
        real_word_without_space = word.replace('\s', '')
        real_char_nums = len(real_word_without_space)
        input = word_image.copy()
        scale = height_of_box / input.shape[0]
        input = cv2.resize(input, None, fx=scale, fy=scale)

        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                                                   variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            scores, _ = net(img_torch)
        region_scores = scores[0, :, :, 0].cpu().data.numpy()
        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
        bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
        bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2BGR)
        pursedo_bboxes = watershed(input, bgr_region_scores, idx=idx, viz=False)

        _tmp = []
        for i in range(pursedo_bboxes.shape[0]):
            if np.mean(pursedo_bboxes[i].ravel()) > 2:
                _tmp.append(pursedo_bboxes[i])
            else:
                print("filter bboxes", pursedo_bboxes[i])
        pursedo_bboxes = np.array(_tmp, np.float32)
        if pursedo_bboxes.shape[0] > 1:
            index = np.argsort(pursedo_bboxes[:, 0, 0])
            pursedo_bboxes = pursedo_bboxes[index]

        confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))

        bboxes = []
        if confidence <= 0.5:
            width = input.shape[1]
            height = input.shape[0]

            width_per_char = width / len(word)
            for i, char in enumerate(word):
                if char == ' ':
                    continue
                left = i * width_per_char
                right = (i + 1) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)

            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5

        else:
            bboxes = pursedo_bboxes
        bboxes /= scale
        try:
            for j in range(len(bboxes)):
                # ones = np.ones((4, 1))
                # tmp = np.concatenate([bboxes[j], ones], axis=-1)
                # I = np.matrix(MM).I
                # ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                # bboxes[j] = ori[:, :2]

                I = np.matrix(MM).I
                bb = bboxes[j]
                bb = bb.reshape((8))
                x1, y1, x2, y2, x3, y3, x4, y4 = bb
                min_x = min(abs(x1 - x2), abs(x3 - x4))
                min_y = min(abs(y1 - y4), abs(y2 - y3))
                if min_x < expand_small_box:
                    x1 = max(0, x1 - expand_small_box)
                    x2 = min(image.shape[1], x2 + expand_small_box)
                    x3 = min(image.shape[1], x3 + expand_small_box)
                    x4 = max(0, x4 - expand_small_box)
                if min_y < expand_small_box:
                    y1 = max(0, y1 - expand_small_box)
                    y2 = max(0, y2 - expand_small_box)
                    y3 = min(image.shape[0], y3 + expand_small_box)
                    y4 = min(image.shape[0], y4 + expand_small_box)
                bb = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(4, 2)
                bb = bb.reshape(1, bb.shape[0], bb.shape[1])
                bb = cv2.perspectiveTransform(bb, I)
                bboxes[j] = bb.reshape((4, 2))

        except Exception as e:

            print(e, word_bbox, word)

        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)

        return bboxes, region_scores, confidence

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def get_imagename(self, index):
        return None

    def saveInput(self, imagename, image, region_scores, affinity_scores, confidence_mask):

        boxes, polys = craft_utils.getDetBoxes(region_scores / 255, affinity_scores / 255, 0.7, 0.4, 0.4, False)
        boxes = np.array(boxes, np.int32) * 2
        if len(boxes) > 0:
            np.clip(boxes[:, :, 0], 0, image.shape[1])
            np.clip(boxes[:, :, 1], 0, image.shape[0])
            for box in boxes:
                cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))
        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        gt_scores = np.hstack([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color])
        confidence_mask_gray = np.hstack([np.zeros_like(confidence_mask_gray), confidence_mask_gray])
        output = np.concatenate([gt_scores, confidence_mask_gray],
                                axis=0)
        output = np.hstack([image, output])
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + '/output'), "%s_input.jpg" % imagename)
        print(outpath)
        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

    def saveImage(self, imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask):
        output_image = np.uint8(image.copy())
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if len(bboxes) > 0:
            affinity_bboxes = np.int32(affinity_bboxes)
            for i in range(affinity_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
            for i in range(len(bboxes)):
                _bboxes = np.int32(bboxes[i])
                for j in range(_bboxes.shape[0]):
                    cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        heat_map = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=1)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)
        outpath = os.path.join(os.path.join(os.path.dirname(__file__) + '/output'), imagename)

        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

    def pull_item(self, index):
        image, character_bboxes, words, confidence_mask, confidences = self.load_image_gt_and_confidencemask(index)
        print(self.get_imagename(index))
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []

        if len(character_bboxes) > 0:
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,
                                                                                          character_bboxes,
                                                                                          words,image)
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
        # showmat("img",image)
        # showmat("reg",region_scores)
        # showmat("aff",affinity_scores)
        # showmat("conf",confidence_mask)
        # score = np.hstack((region_scores, affinity_scores))
        # cv2.imwrite("debug/" + self.get_imagename(index), score)
        # cv2.imwrite("debug/" + "reg_" + self.get_imagename(index), region_scores)
        if self.viz:
            self.saveImage(self.get_imagename(index), image.copy(), character_bboxes, affinity_bboxes, region_scores,
                           affinity_scores,
                           confidence_mask)
        random_transforms = [image, region_scores, affinity_scores, confidence_mask]
        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size))
        # showmat("img random_crop", random_transforms[0])
        # showmat("regrandom_crop", random_transforms[1])
        # showmat("affrandom_crop", random_transforms[2])
        # showmat("confrandom_crop", random_transforms[3])
        random_transforms = random_horizontal_flip(random_transforms)
        # showmat("imgrandom_horizontal_flip", random_transforms[0])
        # showmat("regrandom_horizontal_flip", random_transforms[1])
        # showmat("affrandom_horizontal_flip", random_transforms[2])
        # showmat("confrandom_horizontal_flip", random_transforms[3])
        random_transforms = random_rotate(random_transforms)
        # showmat("imgrandom_rotate", random_transforms[0])
        # showmat("regrandom_rotate", random_transforms[1])
        # showmat("affrandom_rotate", random_transforms[2])
        # showmat("confrandom_rotate", random_transforms[3])

        cvimage, region_scores, affinity_scores, confidence_mask = random_transforms
        # print(cvimage.shape)
        # showmat("img trans", cvimage)
        # showmat("reg trans", region_scores)
        # showmat("aff trans", affinity_scores)
        # showmat("conf trans",confidence_mask)

        region_scores = self.resizeGt(region_scores)
        affinity_scores = self.resizeGt(affinity_scores)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.viz:
            self.saveInput(self.get_imagename(index), cvimage, region_scores, affinity_scores, confidence_mask)
        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        # image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores / 255).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        return image, region_scores_torch, affinity_scores_torch, confidence_mask_torch, confidences


class CharLevelDataset(craft_base_dataset):
    def __init__(self, folder, target_size=768
                 , viz=False, debug=False):
        super(CharLevelDataset, self).__init__(target_size, viz, debug)
        self.folder = folder
        self.imagenames = self.process_multi_folder(folder)
        self.images_path = []
        for imagename in self.imagenames:
            self.images_path.append(os.path.basename(imagename))

    def process_multi_folder(self, list_folders):
        all_imgs = []
        for folder in list_folders:
            img_folder = os.path.join(folder, 'imgs')
            gt_files = os.listdir(os.path.join(folder, "gt"))
            img_files = glob.glob(os.path.join(img_folder, "*"))
            for img_file in img_files:
                img_name_without_ext = os.path.splitext(os.path.basename(img_file))[0]
                gt_fname = "gt_{}.txt".format(img_name_without_ext)
                if gt_fname in gt_files:
                    all_imgs.append(img_file)
        return all_imgs

    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(coor) for coor in ori_box]
            word = len(box) // 8
            word = 'A' * word
            # print(word)
            box = np.array(box, np.int32).reshape(-1, 4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
                continue
            if len(word.strip()) == 0:
                continue
            bboxes.append(box)
            words.append(word)
        return bboxes, words

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]

    def load_image_gt_and_confidencemask(self, index):
        imagename = self.imagenames[index]
        file_name = os.path.basename(imagename)
        image = cv2.imread(imagename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        folder = Path(imagename).parents[1]
        gt_folder = os.path.join(folder, "gt")
        # print(imagename)
        gt_path = os.path.join(gt_folder, "gt_%s.txt" % os.path.splitext(file_name)[0])
        word_bboxes, words = self.load_gt(gt_path)
        character_bboxes = []
        confidences = []
        for i in range(len(words)):
            bboxes = word_bboxes[i]
            assert (len(bboxes) == len(words[i]))
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences


class Synth80k(craft_base_dataset):

    def __init__(self, synthtext_folder, target_size=768, viz=False, debug=False):
        super(Synth80k, self).__init__(target_size, viz, debug)
        self.synthtext_folder = synthtext_folder
        gt = scio.loadmat(os.path.join(synthtext_folder, 'gt.mat'))
        self.charbox = gt['charBB'][0]
        self.image = gt['imnames'][0]
        self.imgtxt = gt['txt'][0]

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.imgtxt)

    def get_imagename(self, index):
        return self.image[index][0]

    def load_image_gt_and_confidencemask(self, index):
        img_path = os.path.join(self.synthtext_folder, self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        image = random_scale(image, _charbox, self.target_size)
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences


class WordLevelDataset(craft_base_dataset):
    def __init__(self, net, icdar2013_folder, target_size=768, viz=False, debug=False):
        super(WordLevelDataset, self).__init__(target_size, viz, debug)
        self.net = net
        self.imagenames = self.process_multi_folder(icdar2013_folder)
        self.images_path = []
        for imagename in self.imagenames:
            self.images_path.append(os.path.basename(imagename))

    def process_multi_folder(self, list_folders):
        all_imgs = []
        for folder in list_folders:
            img_folder = os.path.join(folder, 'imgs')
            gt_files = os.listdir(os.path.join(folder, "gt"))
            img_files = glob.glob(os.path.join(img_folder, "*"))
            for img_file in img_files:
                img_name_without_ext = os.path.splitext(os.path.basename(img_file))[0]
                gt_fname = "gt_{}.txt".format(img_name_without_ext)
                if gt_fname in gt_files:
                    all_imgs.append(img_file)
        return all_imgs

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]

    def check_valid_box(self, box):
        (tl, tr, br, bl) = box
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        if widthA < 1 or widthB < 1 or heightA < 1 or heightB < 1:
            return False
        return True

    def load_image_gt_and_confidencemask(self, index):
        imagename = self.imagenames[index]
        file_name = os.path.basename(imagename)
        folder = Path(imagename).parents[1]
        gt_folder = os.path.join(folder, "gt")
        # print(imagename)
        gt_path = os.path.join(gt_folder, "gt_%s.txt" % os.path.splitext(file_name)[0])
        word_bboxes, words = self.load_gt(gt_path)
        word_bboxes = np.float32(word_bboxes)

        image = cv2.imread(imagename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if random.random() < 0.3:
            jitter = color_jitter_image()
            image = jitter(transforms.ToPILImage()(image))
            image = np.array(image)
        image = random_scale(image, word_bboxes, self.target_size)

        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        character_bboxes = []
        new_words = []
        confidences = []
        height_of_box = config.height_of_box
        expand_small_box = config.expand_small_box
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0:
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
                elif not self.check_valid_box(word_bboxes[i]):
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))

            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0 or not self.check_valid_box(word_bboxes[i]):
                    continue

                pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net, image,
                                                                                               word_bboxes[i],
                                                                                               words[i],
                                                                                               viz=self.viz, idx=i,
                                                                                               height_of_box=height_of_box,
                                                                                               expand_small_box=expand_small_box)
                confidences.append(confidence)
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                # print("confidence: ",confidence)
                # showmat("confi",confidence_mask)
                new_words.append(words[i])
                character_bboxes.append(pursedo_bboxes)
        # for bboxes in character_bboxes:
        #     for box in bboxes:
        #         cv2.fillPoly(image,[np.int32(box)],(random.randint(0,100),random.randint(0,100),random.randint(0,100)))
        #     showmat("box",image)
        return image, character_bboxes, new_words, confidence_mask, confidences

    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[9:]
            word = ','.join(word)
            # print(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
                continue
            if len(word.strip()) == 0:
                continue
            bboxes.append(box)
            words.append(word)
        return bboxes, words


if __name__ == '__main__':
    char=CharLevelDataset(["/home/aimenext/cuongdx/ufj/data/split/train/mixup/char/done"])
    net = CRAFT()
    pre_trained="/home/aimenext/cuongdx/craft/models/pretrained/300_1.00000.pth"
    net.load_state_dict(copyStateDict(torch.load(pre_trained)))
    model_name = os.path.basename(pre_trained)

    net = net.cuda()
    net.eval()
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    # word = WordLevelDataset(net,["/home/aimenext/cuongdx/ufj/data/split/train/mixup/char/done/word"])
    data_loader = torch.utils.data.DataLoader(
        char,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    for index, (image, mask, _, _, _) in enumerate(data_loader):
        print(index)
