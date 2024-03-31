from src.person_detect.utils.datasets import *
from src.person_detect.utils.utils import *


class PersonDetectAPI(object):
    def __init__(self, cfg, device):
        self.opt = cfg.PERSON_DETECT
        self.img_size = self.opt.img_size
        # Initialize
        self.device = device
        # Load model
        self.model = torch.load(self.opt.weights, map_location=self.device)['model']
        #
        self.model.to(self.device).eval()

        # Half precision
        self.half = self.opt.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()
            # Get names and colors

        img = torch.zeros((1, 3,  self.img_size,  self.img_size), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img.float()) if self.device != 'cpu' else None  # run once
        self.names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names

    def predict(self, img0):
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=self.opt.augment)[0]
        # to float
        if self.half:
            pred = pred.half()

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                   fast=True, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        bboxes = []
        labels = []
        scores = []
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    if int(cls) != 0:
                        continue
                    label = self.names[int(cls)]
                    labels.append(label)
                    # label = '%s %.2f' % (self.names[int(cls)], conf)
                    x1, y1, x2, y2 = xyxy
                    bboxes.append([int(x1), int(y1), int(x2), int(y2)])

                    scores.append(float(conf))
        return bboxes, labels, scores

    def crop_human(self, frame, locations):
        human_image = []
        for loc in locations:
            if len(loc) != 0:
                bottom, left, top, right = loc
                sub_frame = frame[int(left):int(right), int(bottom):int(top), :]  # 3 channel image
                human_image.append(sub_frame)
        return human_image
