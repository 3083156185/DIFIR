# encoding: utf-8
from __future__ import division

import numpy as np
from scipy.spatial.distance import cdist

from src.deep_sort import DeepSort
# from src.person_detect_api import PersonDetectAPI
from src.person_detect.person_detect_api import PersonDetectAPI
# from src.reid.api import PersonReIDAPI
# from src.reid.get_video_human import get_video_human
from src.sort import  *
from src.pose.pose_estimate import SppFastPose
import torch
class PersonEngine(object):
    def __init__(self, config, GPUID):
        # 初始化检测模型
        self.detect_model = PersonDetectAPI(cfg=config, device=GPUID)

        # 初始化deep sort跟踪算法
        self.tracker = DeepSort(config=config, device=GPUID)

        #初始化sort跟踪器
        self.sort_tracker = Sort()

        # POSE MODEL.
        self.pose_model = SppFastPose(config, device=GPUID)

    def detect_and_track_deepsort_old(self, frame,  h, w):
        """
        检测，deep sort 跟踪
        :param frame:
        :return:
        """
        # 检测
        bboxes, labels, scores = self.detect_model.predict(frame)
        # deep sort跟踪
        # if len(bboxes):
        #     # Predict skeleton pose of each bboxs.
        #     bboxes = torch.tensor(bboxes, dtype=torch.float32)
        #     scores = torch.tensor(scores, dtype=torch.float32)
        #     poses = self.pose_model.predict(frame, bboxes, scores)
        #     return poses
        # else:
        #     return None
        # deep sort跟踪
        if len(bboxes):
            # Predict skeleton pose of each bboxs.
            # poses = self.pose_model.predict(frame, bboxes, scores)
            track_bboxs, track_ids, track_dict = self.tracker.update(bboxes, scores, frame,  h, w)
            if len(track_bboxs):
                track_bboxs = torch.tensor(track_bboxs, dtype=torch.float32)
                scores = torch.tensor(scores, dtype=torch.float32)
                poses = self.pose_model.predict(frame, track_bboxs, scores)
            else:
                return []
        else:
            poses = []
            # track_bboxs, track_ids, track_dict= [], [], {}
        # return track_bboxs, track_ids, track_dict
        return poses

    def detect_and_track_deepsort(self, frame,  h, w):
        """
        检测，deep sort 跟踪
        :param frame:
        :return:
        """
        # 检测
        bboxes, labels, scores = self.detect_model.predict(frame)
        # deep sort跟踪
        # if len(bboxes):
        #     # Predict skeleton pose of each bboxs.
        #     bboxes = torch.tensor(bboxes, dtype=torch.float32)
        #     scores = torch.tensor(scores, dtype=torch.float32)
        #     poses = self.pose_model.predict(frame, bboxes, scores)
        #     return poses
        # else:
        #     return None
        # deep sort跟踪
        if len(bboxes):
            # Predict skeleton pose of each bboxs.
            # poses = self.pose_model.predict(frame, bboxes, scores)
            track_bboxs, track_ids, track_dict = self.tracker.update(bboxes, scores, frame,  h, w)
            if len(track_bboxs):
                track_bboxs = torch.tensor(track_bboxs, dtype=torch.float32)
                scores = torch.tensor(scores, dtype=torch.float32)
                track_ids = torch.tensor(track_ids, dtype=torch.float32)
                poses = self.pose_model.predict(frame,track_bboxs, track_ids)
            else:
                return []
        else:
            poses = []
            # track_bboxs, track_ids, track_dict= [], [], {}
        # return track_bboxs, track_ids, track_dict
        return poses

    def detect_and_track_sort(self, frame):
        """
        检测，sort跟踪
        :param frame:
        :return:
        """
        # 检测
        bboxes, labels, scores = self.detect_model.predict(frame)

        # sort 跟踪
        dets_to_sort = np.empty((0, 5))
        if len(bboxes):
            for box, conf in zip (bboxes, scores):
                x1, y1, x2, y2 = box
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf])))
            tracked_dets = self.sort_tracker.update(dets_to_sort)
        else:
            tracked_dets = []
        return tracked_dets

    def detect(self, frame):
        """
        检测，跟踪
        :param frame:
        :return:
        """
        bboxes, labels, scores = self.detect_model.predict(frame)

        return bboxes, labels, scores




