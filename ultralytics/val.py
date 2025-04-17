import sys
from pathlib import Path

# 将项目根目录插入到sys.path的最前端
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ultralytics import YOLO
import matplotlib
# Load the model
matplotlib.use('TKAgg')

if __name__ == '__main__':
    model = YOLO('E:/bigchuang/runs/detect/train2_SE+ASPP_250/weights/best.pt')

# Run the evaluation
    results = model.val(data="E:/bigchuang/data/test.yaml")

# # Print specific metrics
    # print("Class indices with average precision:", results.ap_class_index)
    # print("Average precision for all classes:", results.box.all_ap)
    # print("Mean average precision at IoU=0.50:", results.box.map50)
    # print("Mean recall:", results.box.mr)