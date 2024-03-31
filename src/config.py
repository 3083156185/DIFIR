from easydict import EasyDict

config = EasyDict()

config.PERSON_DETECT = EasyDict()
config.PERSON_DETECT.weights = './weights/yolov5s.pt'
config.PERSON_DETECT.half = False  ###max boxes
config.PERSON_DETECT.conf_thres = 0.2
config.PERSON_DETECT.iou_thres = 0.5
config.PERSON_DETECT.img_size = 640
config.PERSON_DETECT.classes = None
config.PERSON_DETECT.agnostic_nms = False
config.PERSON_DETECT.augment = False

config.TRACK = EasyDict()
config.TRACK.weights = './weights/ckpt.t7'
config.TRACK.max_dist = 0.2
config.TRACK.min_confidence = 0.3
config.TRACK.nms_max_overlap = 0.5
config.TRACK.max_iou_distance = 0.7
config.TRACK.max_age = 70
config.TRACK.n_init = 3
config.TRACK.nn_budget = 100

config.POSE = EasyDict()
config.POSE.model_path = './weights/sppe/fast_res50_256x192.pth'


config.PATH = "85720119-1-192.mp4"
config.HOST = "127.0.0.1"
config.PORT = 5001
config.SQUARE = [(154, 122), (443, 122), (443, 279), (154, 279)]