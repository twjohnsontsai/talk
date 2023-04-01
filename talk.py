'''
Author: twjohnsontsai twjohnsontsai@icloud.com
Date: 2023-03-27 12:00:07
LastEditors: twjohnsontsai twjohnsontsai@icloud.com
LastEditTime: 2023-04-01 00:54:40
FilePath: /test/talk.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import dlib
import numpy as np

# 获取当前脚本所在目录的绝对路径
base_path = os.path.dirname(os.path.abspath(__file__))

# 加载面部关键点检测器模型
predictor_path = os.path.join(base_path, "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

# 加载人脸检测器模型
detector = dlib.get_frontal_face_detector()

# 读取目标照片和视频
img = cv2.imread(os.path.join(base_path, "source.png"))
cap = cv2.VideoCapture(os.path.join(base_path, "driving.mp4"))

# 获取目标照片的面部关键点
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray_img, 0)
if len(rects) == 0:
    print("No face detected in target image")
    exit()
face_landmarks = predictor(gray_img, rects[0]).parts()

# 将面部关键点转换为numpy数组
points = []
for p in face_landmarks:
    points.append([p.x, p.y])
points = np.array(points)

# 获取视频文件的一些参数
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 初始化输出视频文件的相关参数
out_video_path = os.path.join(base_path, "result.mp4")
out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (img.shape[1], img.shape[0]))

# 处理视频每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 获取当前帧的面部关键点
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_frame, 0)
    if len(rects) == 0:
        out_video.write(frame)
        continue
    face_landmarks = predictor(gray_frame, rects[0]).parts()

    # 将面部关键点转换为numpy数组
    points_src = []
    for p in face_landmarks:
        points_src.append([p.x, p.y])
    points_src = np.array(points_src)

    # 进行仿射变换
    M, _ = cv2.estimateAffinePartial2D(points_src, points, method=cv2.LMEDS)
    if M is None:
        out_video.write(frame)
        continue
    face = cv2.warpAffine(frame, M, (img.shape[1], img.shape[0]),
 None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 在目标照片中抠出脸部区域
    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    face_masked = cv2.bitwise_and(img, img, mask=mask)

    # 将脸部区域从目标照片中剪裁出来


# 将脸部区域从目标照片中剪裁出来
    x, y, w, h = cv2.boundingRect(np.array([points]))
    face_region = face_masked[y:y + h, x:x + w]

# 进行仿射变换
    face_region_aligned = cv2.warpAffine(face_region, M, (frame.shape[1], frame.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# 将融合后的脸部区域叠加到原视频帧上
    mask = cv2.bitwise_not(mask)
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
    frame_masked[y:y + h, x:x + w] = cv2.resize(face_region_aligned, (w, h))
    out_video.write(frame_masked)



    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

