import numpy as np
import time
import cv2

#Setting the confident and threshold limits
confid = 0.5
thresh = 0.5

#Grabbing the YOLO Labels
labelsPath = "./yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

#Getting the YOLO weights and configurations
weightsPath = "./yolo-coco/yolov3.weights"
configPath = "./yolo-coco/yolov3.cfg"

q = 0

#Grabbing the input video
video_path = "./videos/sample.mp4"

np.random.seed(42)

#Measuring the distance between two points
def dist_measure(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + 550 / ((point1[1] + point2[1]) / 2) * (point1[1] - point2[1]) ** 2) ** 0.5

#Setting the Threshold
def close_enough(point1, point2):
    d_m = dist_measure(point1, point2)
    mid_point = (point1[1] + point2[1])/2
    if 0 < d_m < 0.20 * mid_point:
        return 1
    else:
        return 0

#Bringing the YOLOv3 model from Darknet implementation
model = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_name = model.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]

#================================================================================
#OpenCV Start here
#================================================================================
video = cv2.VideoCapture(video_path)
writer = None
(W, H) = (None, None)

while True:
    (grabbed, frame) = video.read()
    #Setting the 
    if not grabbed:
        break

    #Setting the Width and Height of each frame
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        q = W

    frame = frame[0:H, 200:q]
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()         #Start frame segmentation
    layerOutputs = model.forward(layer_name)
    end = time.time()           #End frame segmentation

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #Only detect "person" as per the YOLO Algorithm
            if LABELS[classID] == "person":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:
        status = []
        close_pair = []
        center = []
        dist = []
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])

            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = close_enough(center[i], center[j])

                if g == 1:
                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1

        total_person = len(center)
        high_risk = status.count(1)
        no_risk = status.count(0)
        flag = 0

        not_follow = "Not Following"
        follow = "Following"

        for i in idxs.flatten():
            cv2.putText(frame, "COVID-19 Social Distance Monitor", (20, 45), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)            

            total_count = "Total: " + str(total_person)
            follow_count = "Following: " + str(high_risk)
            not_follow_count = "Not Following: " + str(no_risk)

            cv2.putText(frame, total_count, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            cv2.putText(frame, follow_count, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.putText(frame, not_follow_count, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[flag] == 1:
                #Crimson Box for Pedestrian not following SOcial Distancing
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                y = y - 10 if y else y + 10
                cv2.putText(frame, not_follow, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            elif status[flag] == 0:
                #Green Box for Pedestrian following social distancing
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                y = y - 10 if y else y + 10
                cv2.putText(frame, follow, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            flag += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)

        #Window name is COVID-19 Distance Monitor
        cv2.imshow('COVID-19 Distance Monitor', frame)
        key = cv2.waitKey(1) & 0xFF

        #If pressed 'Esc', quit the frame
        if key == 27:
            print("[INFO]: Training abruptly ended")
            break

    #Compiling the frames together to form output.mp4 video
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
print("[DONE]: Saved as output.mp4")
writer.release()
video.release()
