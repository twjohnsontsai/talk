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
        points = np.array([(p.x, p.y) for p in points], dtype=np.int32)

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

    # 将每个五官的半径和中心点保存到一个列表中
        radii = [mouth_radius, nose_radius, eye_radius]
        features = [centers, radii]

    # 循环遍历每一个五官，进行交换
    for i in range(len(features)):
        # 获取要交换的五官
        feature1 = features[i]
        feature2 = features[(i+1) % len(features)]

        # 计算要交换的中心点和半径
        center1, center2 = feature1[0], feature2[0]
        radius1, radius2 = feature1[1], feature2[1]

        # 计算旋转角度和缩放比例
        angle = np.arctan2(center2[1]-center1[1], center2[0]-center1[0])
        scale = np.linalg.norm(radius2) / np.linalg.norm(radius1)

        # 构造仿射变换矩阵
        M = cv2.getRotationMatrix2D(tuple(center1), angle*180/np.pi, scale)

        # 对当前五官进行仿射变换
        feature1_points = np.array(
            [np.dot(M, (x, y, 1)).astype(int)[:2] for x, y in feature1[0]])
        feature1_radius = np.array(radius1) * scale

        # 将交换后的五官绘制在图像上
        cv2.fillPoly(frame, [feature1_points], (0, 0, 0))
        cv2.circle(frame, tuple(feature1[0][0]), int(
            feature1_radius), (0, 0, 255), -1)

# 显示当前帧的图像
        cv2.imshow('frame', frame)

# 按下q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 释放资源
        cap.release()
        cv2.destroyAllWindows()
