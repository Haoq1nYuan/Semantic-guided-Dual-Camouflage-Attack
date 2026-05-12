import os.path
import random
from PIL import Image
from skimage import io
from torchvision import models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import transforms
from ultralytics import YOLO
from modules.dataloader import UpdateDataloader
from modules.renderer import NeuralRenderer
from tools.eval_utils import *
from detectors.yolov5.models.experimental import attempt_load
from detectors.yolov5.utils.general import non_max_suppression
from torchvision.transforms.functional import to_pil_image
from detectors.efficientdet import create_model


class DualDimensionalEvaluator:
    def __init__(self, mode, config):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.config = config

        if mode == "adversarial":
            self.validate_paths([
                config['adv_evaluate']['texture_path'],
                config['dataset']['test_bg_dir'],
                config['dataset']['test_label_dir'],
                config['car_model']['obj_mesh'],
                "3Dmodels/original1.png",
                "3Dmodels/original2.png",
            ])
            transform = transforms.ToTensor()
            resize = transforms.Resize((2048, 2048))

            adv_texture = resize(transform(Image.open(config['adv_evaluate']['texture_path']).convert("RGB"))).permute(1, 2, 0)

            original1 = transform(Image.open("3Dmodels/original1.png")).permute(1, 2, 0)  # [2048, 2048, 3]
            original2 = transform(Image.open("3Dmodels/original2.png")).permute(1, 2, 0)  # [2048, 2048, 3]
            maskUV1 = (original1 >= (250 / 255)).int()
            maskUV2 = (original2 >= (250 / 255)).int()

            resized_adv_texture = resize(adv_texture.permute(2, 0, 1)).permute(1, 2, 0)
            tem_texture = resized_adv_texture * maskUV1 + original1 * (1 - maskUV1)
            self.final_adv_texture = tem_texture * (1 - maskUV2) + original2 * maskUV2
            self.final_adv_texture = self.final_adv_texture.to(self.device)   # UV map + adv_texture

            self.renderer = NeuralRenderer(self.device, "3Dmodels/car.obj")
            self.new_subfolder = ""
        elif mode == "natural":
            self.validate_paths([
                config['nat_evaluate']['texture_path'],
                config['dataset']['nat_bg_dir'],
            ])
            self.padding = config['nat_evaluate']['padding']
        else:
            raise ValueError(f"Unsupported evaluator mode: {mode}")

    @staticmethod
    def validate_paths(paths):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path does not exist: {path}")

    def nat_evaluate(self, is_save):
        if is_save:
            self.new_subfolder = set_save("results/evaluate_results/natural")

        sample_num = self.config['nat_evaluate']['bg_sample_num']

        # generate random mask for adv_texture embedding
        masks = []
        for i in range(sample_num):
            mask = np.zeros((400, 400), dtype=np.uint8)
            side_length = random.randint(20, 50)
            x = random.randint(self.padding, 400 - side_length - self.padding)
            y = random.randint(self.padding, 400 - side_length - self.padding)
            mask[y: y + side_length, x: x + side_length] = 255
            masks.append((mask, side_length))

        SSIM = 0
        FSIM = 0
        CSI = 0

        # random sample
        files = [f for f in os.listdir(self.config['dataset']['nat_bg_dir']) if f.endswith(".jpg")]
        if len(files) < sample_num:
            raise ValueError(
                f"Naturalness evaluation needs at least {sample_num} .jpg files in "
                f"{self.config['dataset']['nat_bg_dir']}, found {len(files)}."
            )
        idx_list = random.sample(range(len(files)), sample_num)
        for idx in range(len(idx_list)):
            bg = io.imread(os.path.join(self.config['dataset']['nat_bg_dir'], f"e{idx_list[idx] + 1:03}.jpg"))  # original bg
            bg = cv2.resize(bg, (400, 400))
            tem_bg = bg.copy()   # bg to be embedded
            texture = io.imread(self.config['nat_evaluate']['texture_path'])[:, :, :3]   # adv_texture

            mask = masks[idx][0]
            size = masks[idx][1]  # the size of the embedded area
            white_area_indices = np.where(mask == 255)
            top_left_y, top_left_x = np.min(white_area_indices[0]), np.min(white_area_indices[1])

            # resize adv_texture and embed it
            texture = cv2.resize(texture, (size, size))
            tem_bg[top_left_y: top_left_y + size, top_left_x: top_left_x + size, :] = texture

            # extract regions of interest (containing texture regions and their surroundings)
            # extract the same regions from the original background and the modified background for comparison
            bg_roi = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)[top_left_y - self.padding: top_left_y + size + self.padding,
                     top_left_x - self.padding: top_left_x + size + self.padding, :]
            tem_bg_roi = cv2.cvtColor(tem_bg, cv2.COLOR_BGR2RGB)[top_left_y - self.padding: top_left_y + size + self.padding,
                       top_left_x - self.padding: top_left_x + size + self.padding, :]

            if is_save:
                cv2.imwrite(os.path.join(self.new_subfolder, f"{idx}_bg.jpg"), bg_roi)
                cv2.imwrite(os.path.join(self.new_subfolder, f"{idx}_revise.jpg"), tem_bg_roi)

            SSIM += calculate_ssim(bg_roi, tem_bg_roi)
            FSIM += fsim(bg_roi, tem_bg_roi)
            CSI += csi(bg_roi, tem_bg_roi)

        print(f"SSIM: {SSIM / sample_num:.4f}, FSIM: {FSIM / sample_num:.4f}, CSI: {CSI / sample_num:.4f}")

    def adv_evaluate(self, model, is_save):
        items = os.listdir(self.config['dataset']['test_bg_dir'])
        total_data_num = len(items)
        if total_data_num < self.config['adv_evaluate']['render_batch_size']:
            raise ValueError(
                f"Adversarial evaluation needs at least {self.config['adv_evaluate']['render_batch_size']} files in "
                f"{self.config['dataset']['test_bg_dir']}, found {total_data_num}."
            )
        file_number_list = random.sample(range(0, total_data_num), self.config['adv_evaluate']['render_batch_size'])

        img_batch, data_number_list = self.renderer.render(self.final_adv_texture,
                                                           self.config['adv_evaluate']['render_batch_size'],
                                                           file_number_list, self.config['dataset']['test_bg_dir'],
                                                           self.config['adv_evaluate']['seed'])

        renderDataloader = UpdateDataloader(img_batch, self.config['adv_evaluate']['detect_batch_size'])

        if is_save:
            self.new_subfolder = set_save("results/evaluate_results/adversarial")

        if model.startswith("yolov5"):
            self.yolov5_detection(model, is_save, renderDataloader, data_number_list, self.new_subfolder)
        elif model.startswith("yolov8"):
            self.yolov8_detection(model, is_save, renderDataloader, data_number_list, self.new_subfolder)
        elif model.startswith("EfDet"):
            self.EfDet_detection(model, is_save, renderDataloader, data_number_list, self.new_subfolder)
        elif model == "FrRCNN":
            self.FrRCNN_detection(is_save, renderDataloader, data_number_list, self.new_subfolder)
        elif model == "DETR":
            self.DETR_detection(is_save, renderDataloader, data_number_list, self.new_subfolder)

    def yolov5_detection(self, model, is_save, renderDataloader, label_idx, new_subfolder=None):
        model = attempt_load(f'detectors/yolov5/{model}.pt', self.device).eval().to(self.device)

        fail_cnt = 0
        total_cnt = 0
        idx = 0
        for batch in renderDataloader:
            idx += 1
            result = model(batch)[0]
            preds = non_max_suppression(result, conf_thres=0.25, iou_thres=0.45,
                                        classes=self.config['adv_evaluate']['target_class'], agnostic=False, max_det=1000)

            fail_batch_cnt, total_batch_cnt = self.figure_batch_results(batch, preds, total_cnt, label_idx, is_save,
                                                                        new_subfolder, idx)

            fail_cnt += fail_batch_cnt
            total_cnt += total_batch_cnt

        print(f"finished, {fail_cnt}/{total_cnt} detected, {fail_cnt / total_cnt:.2f}% of all pics, "
              f"DSR: {(total_cnt - fail_cnt) / total_cnt:.4f}.")

    def yolov8_detection(self, model, is_save, renderDataloader, label_idx, new_subfolder=None):
        model = YOLO(f"detectors/yolov8/{model}.pt").eval().to(self.device)

        total_cnt = 0
        fail_cnt = 0
        idx = -1
        for batch in renderDataloader:
            idx += 1
            result = model.predict(source=batch, conf=0.25, iou=0.45,
                                   classes=[self.config['adv_evaluate']['target_class']], verbose=False)

            images = []
            preds = []
            for item in result:
                images.append(item.orig_img)
                if item.boxes.shape[0] > 0:
                    xyxy = item.boxes.xyxy  # [N, 4]
                    conf = item.boxes.conf  # [N]
                    cls = item.boxes.cls  # [N]
                    box_tensor = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1)
                    preds.append(box_tensor)
                else:
                    preds.append(torch.empty((0, 6), device=item.boxes.xyxy.device))

            fail_batch_cnt, total_batch_cnt = self.figure_batch_results(batch, preds, total_cnt, label_idx, is_save,
                                                                        new_subfolder, idx)

            fail_cnt += fail_batch_cnt
            total_cnt += total_batch_cnt

        print(f"finished, {fail_cnt}/{total_cnt} detected, {fail_cnt / total_cnt:.2f}% of all pics, "
              f"DSR: {(total_cnt - fail_cnt) / total_cnt:.4f}.")

    def EfDet_detection(self, model, is_save, renderDataloader, label_idx, new_subfolder):
        size = {'EfDetd0': 512, 'EfDetd1': 640, 'EfDetd2': 768}
        size_input = size[model]
        resize_input = transforms.Resize((size_input, size_input))
        resize_output = transforms.Resize((640, 640))

        model = create_model(f'tf_efficientdet_{model[-2:]}', bench_task='predict', num_classes=90, pretrained=True)
        model = model.to(self.device)

        fail_cnt = 0
        total_cnt = 0
        idx = 0
        for batch in renderDataloader:
            idx += 1
            batch = resize_input(batch)
            with torch.no_grad():
                results = model(batch)
            batch = resize_output(batch)

            preds = []
            for result in results:
                box_preds = []
                for box_idx in range(result.shape[0]):
                    box = result[box_idx][:4] / size_input * 640
                    score = result[box_idx][4]
                    cls = result[box_idx][5] - 1  # label of 'car' in EfDet -> 3
                    if cls == self.config['adv_evaluate']['target_class'] and score >= 0.25:
                        box_pred = torch.tensor([box[0], box[1], box[2], box[3], score, cls])
                        box_preds.append(box_pred)
                box_preds = torch.stack(box_preds) if box_preds else torch.tensor([])
                preds.append(box_preds)

            fail_batch_cnt, total_batch_cnt = self.figure_batch_results(batch, preds, total_cnt, label_idx, is_save,
                                                                        new_subfolder, idx)

            fail_cnt += fail_batch_cnt
            total_cnt += total_batch_cnt

        print(f"finished, {fail_cnt}/{total_cnt} detected, {fail_cnt / total_cnt:.2f}% of all pics, "
              f"DSR: {(total_cnt - fail_cnt) / total_cnt:.4f}.")

    def FrRCNN_detection(self, is_save, renderDataloader, label_idx, new_subfolder=None):
        model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
        model = model.to(self.device)

        total_cnt = 0
        fail_cnt = 0
        idx = 0
        for batch in renderDataloader:
            idx += 1
            with torch.no_grad():
                outputs = model(batch)

            preds = []
            for output in outputs:
                car_indices = (output['labels'] == 3) & (output['scores'] >= 0.25)
                box = output['boxes'][car_indices]
                score = output['scores'][car_indices]
                cls = output['labels'][car_indices] - 1
                preds.append(torch.cat([box[:, 0:1], box[:, 1:2], box[:, 2:3], box[:, 3:4],
                                        score.unsqueeze(1), cls.float().unsqueeze(1)], dim=1))

            fail_batch_cnt, total_batch_cnt = self.figure_batch_results(batch, preds, total_cnt, label_idx, is_save,
                                                                        new_subfolder, idx)

            fail_cnt += fail_batch_cnt
            total_cnt += total_batch_cnt

        print(f"finished, {fail_cnt}/{total_cnt} detected, {fail_cnt / total_cnt:.2f}% of all pics, "
              f"DSR: {(total_cnt - fail_cnt) / total_cnt:.4f}.")

    def DETR_detection(self, is_save, renderDataloader, label_idx, new_subfolder=None):
        model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True).eval()
        model = model.to(self.device)

        total_cnt = 0
        fail_cnt = 0
        idx = 0
        for batch in renderDataloader:
            idx += 1
            with torch.no_grad():
                output = model(batch)

            preds = []
            for img_idx in range(batch.shape[0]):
                pred_logits = output['pred_logits'][img_idx][:, :91]
                pred_boxes = output['pred_boxes'][img_idx]
                max_output = pred_logits.softmax(-1).max(-1)
                topk = max_output.values.topk(5)
                pred_logits = pred_logits[topk.indices]
                pred_boxes = pred_boxes[topk.indices]
                box_preds = []
                for logits, box in zip(pred_logits, pred_boxes):
                    cls = logits.argmax()
                    score = logits.softmax(-1)[cls]
                    cls -= 1   # label of 'car' in DETR -> 3
                    if cls == self.config['adv_evaluate']['target_class'] and score >= 0.25:
                        box = box * 640
                        x, y, w, h = box
                        x0, x1 = x - w // 2, x + w // 2
                        y0, y1 = y - h // 2, y + h // 2
                        box_pred = torch.tensor([x0, y0, x1, y1, score, cls.float()])
                        box_preds.append(box_pred)
                box_preds = torch.stack(box_preds) if box_preds else torch.tensor([])
                preds.append(box_preds)

            fail_batch_cnt, total_batch_cnt = self.figure_batch_results(batch, preds, total_cnt, label_idx, is_save,
                                                                        new_subfolder, idx)

            fail_cnt += fail_batch_cnt
            total_cnt += total_batch_cnt

        print(f"finished, {fail_cnt}/{total_cnt} detected, {fail_cnt / total_cnt:.2f}% of all pics, "
              f"DSR: {(total_cnt - fail_cnt) / total_cnt:.4f}.")

    def figure_batch_results(self, batch, preds, detection_idx, label_idx, is_save, new_subfolder, batch_idx):
        total_batch_cnt = 0
        fail_batch_cnt = 0
        images = batch.cpu()
        for i in range(len(images)):
            image = to_pil_image(images[i])

            image_with_boxes, tem_cnt = draw_boxes(image, preds[i], detection_idx, label_idx,
                                                   self.config['dataset']['test_label_dir'],
                                                   self.config['adv_evaluate']['target_class'])
            fail_batch_cnt += tem_cnt
            total_batch_cnt += 1
            detection_idx += 1

            if is_save:
                output_path = f"{new_subfolder}/{detection_idx}_{bool(tem_cnt)}.png"
                cv2.imwrite(output_path, image_with_boxes[:, :, ::-1])

        print(f"Saved batch {batch_idx}, {fail_batch_cnt}/{total_batch_cnt} detected.")

        return fail_batch_cnt, total_batch_cnt
