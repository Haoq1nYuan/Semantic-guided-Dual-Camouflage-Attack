import os
import cv2
import numpy as np
import torch
from skimage.color import rgb2gray, rgb2lab
from skimage.metrics import structural_similarity as ssim
import phasepack
from skimage.metrics import mean_squared_error


def set_save(output_dir):
    subfolders = [f.name for f in os.scandir(output_dir) if f.is_dir()]
    folder_numbers = []
    for folder in subfolders:
        try:
            folder_numbers.append(int(folder))
        except ValueError:
            continue

    if folder_numbers:
        new_folder_number = max(folder_numbers) + 1
    else:
        new_folder_number = 1

    new_subfolder = str(output_dir) + '/' + str(new_folder_number)
    os.mkdir(new_subfolder)

    return new_subfolder


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def draw_boxes(image, pred, cnt, label_idx_list, label_dir, target_class):
    image = np.array(image)

    with open(os.path.join(label_dir, f'data{label_idx_list[cnt]}.txt'), 'r', encoding='utf-8') as file:
        label = list(map(int, file.read().replace(',', '').split(' ')))

    cv2.rectangle(image, (label[0], label[1]), (label[2], label[3]), (0, 0, 255), 1)
    cv2.putText(image, "Label", (label[0], label[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if pred.numel():
        flag = False
        for idx in range(pred.shape[0]):
            if bbox_iou(pred[idx, :4], torch.tensor(label)) > 0.5 and pred[idx, 5] == target_class:
                flag = True

            box = pred[idx, :4].cpu().numpy()
            score = pred[idx, 4].item()
            cls = pred[idx, 5].item()

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            title = f"Cls: {int(cls)}, Conf: {score:.2f}"
            cv2.putText(image, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image, int(flag)
    else:
        return image, 0


def calculate_ssim(img1, img2):
    img1_gray = rgb2gray(img1)
    img2_gray = rgb2gray(img2)

    ssim_value, _ = ssim(img1_gray, img2_gray, data_range=1.0, full=True)

    return ssim_value


def phase_congruency(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pc = phasepack.phasecongmono(gray)
    return pc[0]


def gradient_magnitude(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude


def fsim(img1, img2):
    pc1 = phase_congruency(img1)
    pc2 = phase_congruency(img2)
    gm1 = gradient_magnitude(img1)
    gm2 = gradient_magnitude(img2)

    T1 = 0.85
    T2 = 160

    pc_sim = (2 * pc1 * pc2 + T1) / (pc1 ** 2 + pc2 ** 2 + T1)
    gm_sim = (2 * gm1 * gm2 + T2) / (gm1 ** 2 + gm2 ** 2 + T2)

    fsim_index = np.sum(pc_sim * gm_sim) / np.sum(pc_sim)
    return fsim_index


def csi(img1, img2):
    img1_lab = rgb2lab(img1)
    img2_lab = rgb2lab(img2)

    ssim_values = []
    for i in range(3):
        ssim_val, _ = ssim(img1_lab[..., i], img2_lab[..., i], data_range=1.0, full=True)
        ssim_values.append(ssim_val)

    csi_val = np.mean(ssim_values)
    return csi_val


def mse(image1, image2):
    return mean_squared_error(image1 / 255.0, image2 / 255.0)