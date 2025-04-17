import sys
from pathlib import Path

# 将项目根目录插入到sys.path的最前端
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ultralytics import YOLO

# model = YOLO('yolov8n.pt')
# print(model.info()) 

import torch
import torch.nn as nn

model = YOLO('E:/bigchuang/ultralytics/ultralytics/cfg/models/v8/yolov8n_transformer.yaml', task='detect')
# .load('E:/bigchuang/yolov8n.pt')

if __name__ == '__main__':
    results = model.train(data='E:/bigchuang/data/test.yaml', epochs=500, batch=8, pretrained=False, amp=True, cos_lr=True)

# from ultralytics.nn.modules import SEAttention
# print(SEAttention)  # 应输出类定义信息