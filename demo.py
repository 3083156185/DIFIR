import json
import socket

import cv2
from shapely import geometry
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from api.person_engine import PersonEngine
from src.config import config
from src.utils.common import get_json_data, write_json_data
from src.utils.utils import *
from imutils.video import FileVideoStream, VideoStream
from src.fn import draw_single
from pyskl.apis import inference_recognizer, init_recognizer

import time

json_path = "flag.json"

the_revised_dict = get_json_data(json_path)
write_json_data(json_path, the_revised_dict)
GPUID = the_revised_dict["GPUID"]

person_engine = PersonEngine(config=config, GPUID="cuda:{}".format(GPUID))
# person_engine = PersonEngine(config=config, GPUID="cpu")

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    color = compute_color_for_labels(int(label+1))
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize('ID-{}'.format(label), 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, 'ID-{}'.format(label), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# xyxy2tlwh函数  这个函数一般都会自带
def xyxy2tlwh(x):
    '''
    (top left x, top left y,width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def detect_track_sort(url):
    """
    :param url: 待识别的视频
    """
    action_config = "weights/action/slowonly_r50_hmdb51_k400p/s1_joint.py"
    action_checkpoint = "weights/action/best_top1_acc_epoch_4.pth"
    action_device = "cuda:0"
    action_model = init_recognizer(action_config, action_checkpoint, action_device)

    img_shape = (480, 853)
    original_shape = (480, 853)
    num_frame = 25

    # 读取视频
    if os.path.isfile(url):
        vs = FileVideoStream(url).start()
    else:
        vs = VideoStream(src=url).start()

    # 获取视频帧率
    cap = cv2.VideoCapture(url)
    fps = int(cap.get(5))

    t = int(1000 / fps)
    name = 'demo'
    videoWriter = True


    keypoint_result = {}

    # Load label_map
    label_map_path = "weights/action/custom2.txt"
    label_map = [x.strip() for x in open(label_map_path).readlines()]
    print("label_map:{}".format(label_map))
    num = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (1920, 1080))
        (h, w, _) = frame.shape
        """
        final_result.append({
            'bbox': bboxs_pick[j],
            'track_id': track_ids_pick[j],
            'keypoints': merge_pose - 0.3,
            'kp_score': merge_score,
            'proposal_score': torch.mean(merge_score)+ 1.25 * max(merge_score)
        })
        """

        results = person_engine.detect_and_track_deepsort(frame, h, w)

        if len(results):
            for result in results:
                if str(result['track_id'].numpy()) not in keypoint_result.keys():
                    # keypoint_result[str(result['track_id'].numpy())] = deque(maxlen=25)
                    keypoint_result[str(result['track_id'].numpy())] = []
                else:
                    # keypoint_result[str(result['track_id'].numpy())].append(result['keypoints'].numpy())
                    keypoint_result[str(result['track_id'].numpy())].append(
                        {
                            'bbox': result['bbox'],
                            'keypoints': result['keypoints'],
                            'kp_score': result['kp_score'],
                        }

                    )


        if keypoint_result.keys():
            for key in keypoint_result.keys():
                if len(keypoint_result[key]) >= 25:

                    fake_anno = dict(
                        frame_dir='',
                        label=-1,
                        img_shape=img_shape,
                        original_shape=original_shape,
                        start_index=0,
                        modality='Pose',
                        total_frames=num_frame)

                    points = []
                    point_scores = []
                    for kp in keypoint_result[key][-25:]:
                        points.append(kp['keypoints'].numpy())
                        # print(kp['kp_score'].numpy().reshape((1, 17)))
                        point_scores.append(kp['kp_score'].numpy().reshape((1, 17))[0])
                    points = np.array([points]).astype(np.float16)
                    point_scores = np.array([point_scores]).astype(np.float16)
                    fake_anno['keypoint'] = points
                    fake_anno['keypoint_score'] = point_scores
                    # print("fake_anno:{}".format(fake_anno))
                    results = inference_recognizer(action_model, fake_anno)
                    print("id:{}".format(key),results)
                    action_label = label_map[results[0][0]]
                    print("id:{} action_label:{}".format(key, action_label))


                    frame = draw_single(frame, np.concatenate((keypoint_result[key][-1]['keypoints'].numpy(),
                                                               keypoint_result[key][-1]['kp_score'].numpy()), axis=1))

                    bbox = keypoint_result[key][-1]['bbox'].numpy()
                    frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    frame = cv2.putText(frame, str(action_label), (int(bbox[0]), int(bbox[1]-10)),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        0.6, (0, 0, 255), 1)


        # # 将结果保存为视频
        # if videoWriter :
        #     fourcc = cv2.VideoWriter_fourcc(
        #         'm', 'p', '4', 'v')  # opencv3.0
        #     videoWriter = cv2.VideoWriter(
        #         '2_walk_1_turn.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))

        cv2.imshow(name, frame)
        cv2.imwrite("output/3walk2/{}.png".format(num), frame)
        num +=1
        # videoWriter.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if keypoint_result.keys():
        print(keypoint_result.keys())
        # for key in keypoint_result.keys():
        #     print(keypoint_result[key])
    cap.release()
    # videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    with torch.no_grad():
        # url = config.PATH345612
        url = "datasets/1人行走+2人看手机.mp4"
        detect_track_sort(url)
