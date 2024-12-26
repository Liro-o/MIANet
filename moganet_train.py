from ultralytics import YOLO
import warnings


warnings.filterwarnings('ignore')
model = YOLO(r"F:/lic_files/MIANet/ultralytics/cfg/models/MogaNet.yaml")  # 初始化模型
# model = YOLO(r"F:/lic_files/MogaNet/runs/detect/train109/weights/best.pt")  # 初始化模型

model.train(data=r"F:/lic_files/MIANet/DIOR.yaml",
            batch=16,
            epochs=200,
            amp=False,
            workers=0,
            optimizer='AdamW',  # Optimizer
            # cos_lr=True,  # Cosine LR Scheduler
            lr0=0.001,
            device='0'
            )  # 训练

#     ############## 这是val和predict的代码 ##############
# model = YOLO(r"F:/lic_files/MogaNet/runs/detect/train88/weights/best.pt")
#     # model.val(data=r"ultralytics/cfg/datasets/VEDAI.yaml", batch=1, save_json=True, save_txt=False)  # 验证
# model.predict(source=r"H:/show_data", save=True)  #   检测