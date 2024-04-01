import torch

from .main_fast_inference import InferenNet_fastRes50
from .pPose_nms import pose_nms, pose_nms_new
from .utils.eval import getPrediction
from .utils.img import crop_dets


class SppFastPose(object):
    def __init__(self, config, device='cuda'):
        self.inp_h = 256
        self.inp_w = 192
        self.device = device
        self.model_path = config.POSE.model_path
        self.model = InferenNet_fastRes50(model_path=self.model_path).to(device)
        self.model.eval()

    def predict(self, image, bboxs, track_ids):
        inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
        pose_hm = self.model(inps.to(self.device)).cpu().data

        # Cut eyes and ears.
        # pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)

        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
        result = pose_nms(bboxs, track_ids, xy_img, scores)
        # result = pose_nms_new(bboxs, xy_img, scores)
        return result
