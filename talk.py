'''
Author: twjohnsontsai twjohnsontsai@icloud.com
Date: 2023-03-27 12:00:07
LastEditors: twjohnsontsai twjohnsontsai@icloud.com
LastEditTime: 2023-04-05 14:34:01
FilePath: /talk/talk.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import dlib
import cv2
import numpy as np

# 加载人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "/Users/caijionghui/Desktop/test/shape_predictor_68_face_landmarks.dat")

# 读取照片和视频文件
img = cv2.imread("/Users/caijionghui/Desktop/test/source.png")
cap = cv2.VideoCapture("/Users/caijionghui/Desktop/test/driving.mp4")

# 定义五官交换的点
mouth_points = list(range(48, 61))
nose_points = list(range(27, 36))
left_eye_points = list(range(42, 48))
right_eye_points = list(range(36, 42))
jaw_points = list(range(0, 17))

# 将照片转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用人脸检测器检测人脸
rects = detector(gray_img, 0)

# 循环遍历每一个人脸
for rect in rects:
    # 使用关键点检测器检测关键点
    shape = predictor(gray_img, rect)
    points = shape.parts()

    # 将关键点转换为numpy数组
    points = np.array([(p.x, p.y) for p in points])

    # 分别提取五官的关键点
    mouth = points[mouth_points]
    nose = points[nose_points]
    left_eye = points[left_eye_points]
    right_eye = points[right_eye_points]
    jaw = points[jaw_points]

    # 计算每个五官的中心点
    mouth_center = np.mean(mouth, axis=0)
    nose_center = np.mean(nose, axis=0)
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)

    # 将每个五官的中心点保存到一个列表中
    centers = [mouth_center, nose_center, left_eye_center, right_eye_center]

    # 计算每个五官的半径
    mouth_radius = int(np.linalg.norm(mouth[6] - mouth[0]))
    nose_radius = int(np.linalg.norm(nose[4] - nose[0]))
    eye_radius = int(np.linalg.norm(left_eye[3] - left_eye[0]))

# 循环遍历视频的每一帧
while cap.isOpened():
    ret, frame = cap.read()

    # 如果读取视频失败，则退出循环
    if not ret:
        break

    # 将图像转换为灰度图
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测人脸
    rects = detector(gray_frame, 0)

    # 循环遍历每一个人脸
    for rect in rects:
        # 使用关键点检测器检测关键点
        shape = predictor(gray_frame, rect)
        points = shape.parts()

        # 将关键点转换为numpy数组
        points = np.array([(p.x, p.y) for p in points])
        # 省略代码......

    # 将处理后的图像展示出来
    cv2.imshow('result', frame)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获器和所有窗口
cap.release()
cv2.destroyAllWindows()
