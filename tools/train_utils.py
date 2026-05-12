import logging
import os
import re
import sys
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image


def extract_number(file_path):
    file_name = os.path.basename(file_path)
    match = re.search(r'data(\d+)', file_name)
    return int(match.group(1)) if match else 0


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


def set_log(new_subfolder, phase):
    writer = SummaryWriter(os.path.join(new_subfolder, 'tb'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d -> %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(new_subfolder, f"{phase}_{timestamp}.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info(f"Results & logs saved as: {new_subfolder}")

    return writer, logger


def draw_boxes(image, pred):
    box = pred[:4].detach().cpu().numpy()
    score = pred[4].item()
    cls = pred[5].item()

    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    label = f"Cls: {int(cls)}, Conf: {score:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


def draw_detection_results(save_path, img, pred, batch_size, labels, iteration=-1):
    offset = 0
    if iteration != -1:
        new_subfolder = str(save_path) + '/detection/iter_' + str(iteration + 1)
        if not os.path.exists(new_subfolder):
            os.mkdir(new_subfolder)
        else:
            offset = batch_size
    else:
        new_subfolder = str(save_path) + '/best/detection'
        os.mkdir(new_subfolder)

    img = img.cpu()
    for idx in range(len(pred)):
        image = to_pil_image(img[idx])
        image = np.array(image)

        cv2.rectangle(image, (labels[idx][0], labels[idx][1]), (labels[idx][2], labels[idx][3]), (0, 0, 255), 1)
        cv2.putText(image, "Label", (labels[idx][0], labels[idx][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

        if pred[idx][4]:
            flag = True
            image = draw_boxes(image, pred[idx])[:, :, ::-1]
        else:
            flag = False
            image = image[:, :, ::-1]

        output_path = f"{new_subfolder}/{idx + offset}_{flag}.png"
        cv2.imwrite(output_path, image)


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


def save_texture(epoch, folder, adv_texture, perturbation=None):
    if perturbation is None:
        adv_texture = (adv_texture * 255).round().to(torch.uint8)
        adv_texture = Image.fromarray(adv_texture.cpu().numpy(), 'RGB')
        adv_texture.save(os.path.join(folder, f"phase2_adv_texture_epoch_{epoch + 1}.png"))
    else:
        adv_texture = (adv_texture * 255).round().to(torch.uint8)
        adv_texture = Image.fromarray(adv_texture.cpu().numpy(), 'RGB')
        adv_texture.save(os.path.join(folder, f"phase1_adv_texture_epoch_{epoch + 1}.png"))
        perturbation = (perturbation * 255).round().to(torch.uint8)
        perturbation = Image.fromarray(perturbation.cpu().numpy(), 'RGBA')
        perturbation.save(os.path.join(folder, f"starry_texture_Sp_epoch_{epoch + 1}.png"))