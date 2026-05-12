import glob
import random
import time
import argparse
import os
import yaml
from modules.renderer import NeuralRenderer
from modules.dataloader import UpdateDataloader
from modules.generator import SDG
from modules.augmentor import Augmentor
from detectors.yolov5.models.experimental import attempt_load
from detectors.yolov5.utils.general import non_max_suppression
from tools.train_utils import *


def validate_inputs(config):
    required_paths = [
        config['dataset']['train_bg_dir'],
        config['dataset']['train_label_dir'],
        config['dataset']['stg2_idx'],
        config['car_model']['obj_mesh'],
        config['general_train']['detector_dir'],
        config['phase2_train']['spe_texture'],
        config['phase2_train']['perturbation'],
        config['save']['train_output_dir'],
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path}")


def attack_one_step(detectors, augmentor, renderer, generator, attack_class, render_batch_size, detect_batch_size,
                    npz_list, label, device, color_perturbation, dataset_dir):
    # obtain adv_texture for phase2 [texture_size, texture_size, 3]
    adv_texture = generator.gen_phase2_adv_texture(color_perturbation)

    # render img [total_size, 3, 640, 640]
    img_batch, data_number_list = renderer.render(adv_texture, render_batch_size, npz_list, dataset_dir)
    renderDataloader = UpdateDataloader(img_batch, detect_batch_size)

    # run detection
    img_batch_idx = 0
    preds = torch.empty((render_batch_size, 6), device=device)
    imgs = torch.empty((render_batch_size, 3, 640, 640), device=device)
    labels = []
    for img in renderDataloader:
        img = augmentor.apply_random_augmentation(img)

        imgs[img_batch_idx * detect_batch_size: (img_batch_idx + 1) * detect_batch_size] = img

        # obtain detect results [batch_size, 25200, 85]
        detector = random.choice(detectors)
        result = detector(img)[0]
        pred = non_max_suppression(result, conf_thres=0.05, iou_thres=0.45, classes=[attack_class], agnostic=False,
                                   max_det=1000)

        # Match predictions to labels and store results based on IoU threshold
        for pre_idx in range(len(img)):
            labels.append(label[data_number_list[img_batch_idx * detect_batch_size + pre_idx]].tolist())
            if pred[pre_idx].shape[0]:
                flag = False
                for idx in range(pred[pre_idx].shape[0]):
                    if bbox_iou(pred[pre_idx][idx, :4],
                                label[data_number_list[img_batch_idx * detect_batch_size + pre_idx]]) > 0.5:
                        flag = True
                        preds[img_batch_idx * detect_batch_size + pre_idx] = pred[pre_idx][idx]
                        break

                if not flag:
                    preds[img_batch_idx * detect_batch_size + pre_idx] = torch.zeros(6, device=device)
            else:
                preds[img_batch_idx * detect_batch_size + pre_idx] = torch.zeros(6, device=device)

        img_batch_idx += 1

    # calculate classification loss
    closs = preds[:, 4].sum()

    # calculate l2 loss
    l2_loss = torch.norm(color_perturbation)

    return closs, l2_loss, imgs, preds, adv_texture, labels


def SDCA_attack_phase2(config):
    validate_inputs(config)

    # set save
    new_subfolder = set_save(config['save']['train_output_dir'])
    os.mkdir(os.path.join(new_subfolder, "texture"))
    os.mkdir(os.path.join(new_subfolder, "detection"))
    os.mkdir(os.path.join(new_subfolder, "best"))

    # set log
    writer, logger = set_log(new_subfolder, "phase2")

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info("Device: \"{}\"".format(device))

    # load dataset
    npz_list = list(map(int, np.load(config['dataset']['stg2_idx'])['data']))
    # load labels
    txt_files = glob.glob(os.path.join(config['dataset']['train_label_dir'], '*.txt'))
    txt_files_sorted = sorted(txt_files, key=extract_number)
    label = []
    for file_path in txt_files_sorted:
        with open(file_path, 'r', encoding='utf-8') as file:
            label.append(list(map(int, file.read().replace(',', '').split(' '))))
    label = torch.tensor(label)
    logger.info(f"There are {len(npz_list)} data for training")

    # generate random values for the pixel perturbation
    color_perturbation = torch.zeros((config['general_train']['perturbation_size'],
                                      config['general_train']['perturbation_size'], 3),
                                     dtype=torch.float32, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([
        {'params': color_perturbation, 'lr': float(config['phase2_train']['lr'])}
    ], weight_decay=float(config['phase2_train']['weight_decay']))

    # initialize detector
    detectors = [attempt_load(config['general_train']['detector_dir'], "phase2", device).eval()]
    for detector in detectors:
        for param in detector.parameters():
            param.requires_grad = False

    # initialize data augmentor
    augmentor = Augmentor(config['general_train']['brightness_range'], config['general_train']['contrast_range'])

    # initialize renderer
    renderer = NeuralRenderer(device, config['car_model']['obj_mesh'])

    # initialize generator
    generator = SDG(device, config, "phase2")

    # start training
    result_loss = []
    best_epoch = 0
    best_loss = 100
    best_adv_texture = []
    best_imgs = []
    best_preds = []
    best_labels = []

    pre_time = time.perf_counter()

    logger.info("Start training phase2...")
    dataset_dir = config['dataset']['train_bg_dir']
    render_batch_size = config['general_train']['render_batch_size']
    detect_batch_size = config['general_train']['detect_batch_size']
    attack_class = config['general_train']['attack_class']
    iterations = config['phase2_train']['epoch']
    for epoch in range(iterations):
        optimizer.zero_grad()

        closs, l2_loss, imgs, preds, adv_texture, labels = attack_one_step(detectors, augmentor, renderer, generator,
                                                                           attack_class, render_batch_size,
                                                                           detect_batch_size, npz_list, label, device,
                                                                           color_perturbation, dataset_dir)

        # calculate mix loss
        loss = closs + float(config['phase2_train']['alpha']) * l2_loss

        writer.add_scalar('Training/Loss_mix', loss.item(), epoch)
        writer.add_scalar('Training/Loss_cls', closs.item(), epoch)
        writer.add_scalar('Training/Loss_l2', l2_loss.item(), epoch)

        loss.backward()
        optimizer.step()

        # record & save results
        with torch.no_grad():
            result_loss.append(loss.cpu().item())

            # only record the best results when running to the later epoch
            if loss.cpu().item() < best_loss and epoch >= iterations - 50:
                best_epoch = epoch
                best_loss = loss
                best_adv_texture = adv_texture
                best_imgs = imgs
                best_preds = preds
                best_labels = labels

            # save intermediate results
            if epoch % 20 == 0 and config['save']['save_intermediate_results']:
                draw_detection_results(new_subfolder, imgs, preds, detect_batch_size, labels, epoch)
                save_texture(epoch, os.path.join(new_subfolder, 'texture'), adv_texture)

            if epoch % 20 == 0:
                adv_texture_tb = adv_texture.clone().detach().permute(2, 0, 1)
                writer.add_image('Texture/adv_texture', adv_texture_tb, epoch)

            # record time
            end_time = time.perf_counter()
            logger.info(f"[Epoch {epoch + 1}] loss_mix: {loss:.4f}, loss_cls: {closs:.4f}, loss_l2: {l2_loss:.4f}, "
                        f"time: {end_time - pre_time:.2f}s")
            pre_time = end_time

        torch.cuda.empty_cache()

    # save best results
    print(f"Best: Epoch {best_epoch}, loss: {best_loss:.4f}")
    draw_detection_results(new_subfolder, best_imgs, best_preds, detect_batch_size, best_labels)
    save_texture(best_epoch, os.path.join(new_subfolder, 'best'), best_adv_texture)

    writer.close()


DEFAULT_CONFIG = 'config/config_train.yaml'


def parse_args():
    parser = argparse.ArgumentParser(description='Run SDCA phase 2 training.')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Path to the training YAML config.')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    SDCA_attack_phase2(config)


if __name__ == "__main__":
    main()
