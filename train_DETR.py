from ultralytics import RTDETR
import warnings


warnings.filterwarnings('ignore')
model = RTDETR(r"D:/PythonSpace/MogaNet/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")  # 初始化模型

model.train(data=r"D:/PythonSpace/MogaNet/VEDAI.yaml",
            batch=4,
            epochs=200,
            amp=False,
            workers=0,
            optimizer='AdamW',  # Optimizer
            # cos_lr=True,  # Cosine LR Scheduler
            lr0=0.001
            )  # 训练