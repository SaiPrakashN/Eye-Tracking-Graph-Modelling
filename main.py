import cv2
import numpy as np
import dlib
import graphviz

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_PLAIN

up_node_count = down_node_count = left_node_count = right_node_count = center_node_count = 0
graph_edges = []


def write(cv_text):
    cv2.putText(frame, cv_text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


def eye_details(facial_landmarks, eye_points):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                          np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)  # form frame polygon
    cv2.fillPoly(mask, [eye_region], 255)  # filling mask with the polygon
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_part_of_eye = threshold_eye[0:height, 0:int(width / 2)]
    left_white_part = cv2.countNonZero(left_part_of_eye)

    right_part_of_eye = threshold_eye[0:height, int(width / 2):width]
    right_white_part = cv2.countNonZero(right_part_of_eye)

    top_part_of_eye = threshold_eye[0: int(height / 2), 0: width]
    top_white_part = cv2.countNonZero(top_part_of_eye)

    bottom_part_of_eye = threshold_eye[int(height / 2): height, 0: width]
    bottom_white_part = cv2.countNonZero(bottom_part_of_eye)

    return left_white_part, right_white_part, top_white_part, bottom_white_part


centre_ratio, up_ratio_, bot_ratio, left_ratio, right_ratio = [], [], [], [], []

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        right_eye_left, right_eye_right, right_eye_top, right_eye_bottom = eye_details(landmarks,
                                                                                       [42, 43, 44, 45, 46, 47])

        left_eye_left, left_eye_right, left_eye_top, left_eye_bottom = eye_details(landmarks,
                                                                                   [36, 37, 38, 39, 40, 41])

        right_avg = np.average([right_eye_right, left_eye_right])
        left_avg = np.average([right_eye_left, left_eye_left])

        top_avg = np.average([right_eye_top, left_eye_top])
        bot_avg = np.average([right_eye_bottom, left_eye_bottom])

        try:
            top_bot_ratio = top_avg / bot_avg
            right_left_ratio = right_avg / left_avg
        except ZeroDivisionError:
            top_bot_ratio = -1
            right_left_ratio = -1

        gaze_dir_ratio = (right_left_ratio / top_bot_ratio) * np.average([right_left_ratio, top_bot_ratio])

        centre_tot_avg = 4.11859797315494

        up_tot_avg = 2.559399522014517

        left_tot_avg = 9.054994737497642

        right_tot_avg = 1.2054371964460384

        if round(centre_tot_avg) / 2 < round(gaze_dir_ratio) < round(centre_tot_avg) * 2:
            write("Center")
            center_node_count += 1
            graph_edges.append("Center")
        elif round(up_tot_avg) / 2 < round(gaze_dir_ratio) < round(up_tot_avg) * 2:
            write("Up")
            up_node_count += 1
            graph_edges.append("Up")
        elif round(left_tot_avg) / 2 < round(gaze_dir_ratio) < round(left_tot_avg) * 2:
            write("Left")
            left_node_count += 1
            graph_edges.append("Left")
        elif round(right_tot_avg) / 2 < round(gaze_dir_ratio) < round(right_tot_avg) * 2:
            write("Right")
            right_node_count += 1
            graph_edges.append("Right")
        else:
            write("Down")
            down_node_count += 1
            graph_edges.append("Down")

    cv2.imshow("Eye", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32:
        print("\n\n\n__________________________________________________________\n\n\n")

cap.release()
cv2.destroyAllWindows()

dot = graphviz.Digraph()

for i in range(len(graph_edges)):
    try:
        dot.edge(tail_name=graph_edges[i], head_name=graph_edges[i + 1])
    except IndexError:
        pass

path = "resources/Graphs/"

graph_name = raw_input("Enter graph name: ")

dot.render(path+graph_name+'.gv', view=True)
