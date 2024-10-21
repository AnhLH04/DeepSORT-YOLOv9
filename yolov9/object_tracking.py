import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from utils.metrics import box_iou

#config value
video_path = "data_ext/highway.mp4"
conf_threshold = 0.5
tracking_class = 2
mask = cv2.imread('../../DeepSORT/yolov9/data_ext/black.png')
tracker = DeepSort(max_age=20)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = DetectMultiBackend(weights= "weights/yolov9-t-converted.pt")
model = AutoShape(model)
with open("data_ext/classes_names.txt") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size = (len(class_names), 3))
track = []

cap = cv2.VideoCapture(video_path)
count = {'car': 0, 'motorbike': 0, 'person': 0}
track_previous = {}
while True:
    flag, frame = cap.read()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    masked_frame = cv2.bitwise_and(mask, frame)
    cv2.line(frame, (0, 300), (frame.shape[1], 300), (0, 0, 255), 3)
    if not flag:
        break
    results = model(masked_frame)
    #print(results.pred[0].shape)
    detect = []
    class_id_visited = []
    cnt = 0
    results_new = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[0:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)
        for dt_obj in results.pred[0]:
            if (dt_obj != detect_object).all() and box_iou(detect_object[0:4].unsqueeze(0), dt_obj[0:4].unsqueeze(0))[0] > 0.6:
                results_new.append(dt_obj)
        if not any(torch.equal(delete_obj, detect_object) for delete_obj in results_new):
            color = colors[class_id]
            B, G, R = map(int, color)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            #cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            if tracking_class is None:
                if confidence < conf_threshold:
                    continue
            else:
                if tracking_class != class_id or confidence < conf_threshold:
                    continue
            detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])
    tracks = tracker.update_tracks(detect, frame = masked_frame)
    print(len(detect))
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            label = "{}-{}".format(class_names[class_id], track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1-1, y1-20), (x1 + len(label)*12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if track_id in track_previous:
                pre_centroid = track_previous[track_id]
                if(pre_centroid<300 and (y1+y2)/2 >= 300 ) :
                    count[class_names[class_id]]+=1
                    #tracks.pop(track_id, None)
            track_previous[track_id] = (y1+y2) / 2
    cv2.putText(frame, "Car: {}".format(count['car']), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Motor: {}".format(count['motorbike']), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(frame, "Person: {}".format(count['person']), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("OT", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()