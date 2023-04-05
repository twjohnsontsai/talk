import cv2
import numpy as np
import requests
from io import BytesIO
import torch
import torch.backends.cudnn as cudnn
from densepose import DensePosePredictor

# 加载DensePose模型
cudnn.benchmark = True
dp = DensePosePredictor()

# 加载人像照片
response = requests.get('/Users/caijionghui/Desktop/test/source12.png')
img = cv2.imdecode(np.frombuffer(
    BytesIO(response.content).read(), np.uint8), cv2.IMREAD_COLOR)

# 加载人像视频
cap = cv2.VideoCapture('/Users/caijionghui/Desktop/test/driving.mp4')

# 获取视频的宽和高
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 25.0, (width, height))

# 循环处理视频的每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 运行DensePose模型，获取人像的部位和UV图
    dp_out = dp.run_on_opencv_image(frame)

    # 将照片中的人像部分提取出来，resize到和视频帧一样的大小
    mask = np.zeros_like(dp_out['segms'][:, :, 0])
    mask[dp_out['segms'][:, :, 0] == 1] = 1
    y1, x1, y2, x2 = dp_out['rois'][0]
    crop_img = cv2.resize(img[y1:y2, x1:x2], (x2-x1, y2-y1))

    # 将照片中的人像部分替换到视频帧中
    frame_masked = frame.copy()
    frame_masked[mask == 1] = crop_img[mask == 1]

    # 将处理后的视频帧写入输出视频文件
    out.write(frame_masked)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
