水面河道漂浮物检测数据集 4274张，yolo和voc标注
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c9513e7c1ac48eda70cff88537d51be.png)
数据集划分：
训练集3011张，测试集1032张，验证集231张
8类别，标注数量：
bottle - 瓶子：6859
grass - 草：180
branch - 树枝：420
milk-box - 牛奶盒：255
plastic-bag - 塑料袋：392
plastic-garbage - 塑料垃圾：199
ball - 球：46
leaf - 树叶：255
image num - 图像总数量：4274
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8391eba3165848a2ae4dc671f2d12a72.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e84e1b66e62f4d009c3ffe8dbd9f96e4.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1f3feef2a5254fe1a89f4e0a2eea9198.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2265b288af92487e98e1a83fc97006c9.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d749e4e129ec4d67a71698be4a404d8c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d68e92ddf93b412bb3293ca732f2e773.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1fdce628de654b8983e44458bda1ab4a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4eecc0d5be584ffe88102fe120ad660b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/93f77992e920418cbf223674a426d7e5.png)
 **河道漂浮物检测数据集** 的详细表格化描述，以及如何基于 **YOLOv8 构建完整深度学习检测系统** 的全流程指南。

---
以下文字及代码仅供参考学习使用。
## 📊 一、数据集详细表格描述

### ✅ 总体信息

| 项目 | 内容 |
|------|------|
| **总图像数量** | 4,274 张 |
| **图像来源** | 无人机/岸基摄像头航拍或监控（河道、湖泊、水库等水域） |
| **图像分辨率** | 多尺寸（典型为 1920×1080 或 1280×720） |
| **类别数量** | 8 类 |
| **标注格式** | 同时提供 **YOLO 格式（.txt）** 和 **PASCAL VOC 格式（.xml）** |
| **任务类型** | 目标检测（定位并分类水面漂浮垃圾/杂物） |

---

### 🗂️ 数据划分

| 数据集   | 图像数量 | 占比    |
|----------|----------|---------|
| 训练集   | 3,011    | ~70.5%  |
| 验证集   | 231      | ~5.4%   |
| 测试集   | 1,032    | ~24.1%  |
| **总计** | **4,274**| **100%**|

> 💡 注：测试集比例较高，适合用于严格评估模型泛化能力。

---

### 🗑️ 类别分布统计表

| 类别 ID | 英文名称           | 中文名称     | 标注框总数（instances） | 平均每图目标数（估算） | 特点说明 |
|:-------:|--------------------|--------------|--------------------------|------------------------|----------|
| 0       | `bottle`           | 瓶子         | 6,859                    | ≈1.6                   | 最常见，多为塑料瓶、玻璃瓶 |
| 1       | `grass`            | 草           | 180                      | ≈0.04                  | 水面漂浮杂草，易与背景混淆 |
| 2       | `branch`           | 树枝         | 420                      | ≈0.1                   | 细长、不规则形状 |
| 3       | `milk-box`         | 牛奶盒       | 255                      | ≈0.06                  | 利乐包类方形纸盒 |
| 4       | `plastic-bag`      | 塑料袋       | 392                      | ≈0.09                  | 半透明、易变形、反光 |
| 5       | `plastic-garbage`  | 塑料垃圾     | 199                      | ≈0.05                  | 泛指碎片化塑料制品 |
| 6       | `ball`             | 球           | 46                       | ≈0.01                  | 极稀有（如儿童玩具球） |
| 7       | `leaf`             | 树叶         | 255                      | ≈0.06                  | 小目标，密集时成片出现 |
| —       | **总计**           | —            | **8,606**                | —                      | — |

> 🔍 **关键挑战**：
> - **极端类别不平衡**：`bottle` 占比近 80%，而 `ball` 仅 46 个样本。
> - **小目标密集**：树叶、草、塑料碎片常以小尺寸、低对比度出现。
> - **水面干扰**：反光、波纹、阴影易导致误检或漏检。

---

## 🛠️ 二、构建基于 YOLOv8 的河道漂浮物检测系统

### 📁 1. 推荐项目结构

```bash
river_litter_detection/
├── data/
│   ├── images/
│   │   ├── train/      # 3011
│   │   ├── val/        # 231
│   │   └── test/       # 1032
│   ├── labels/         # YOLO .txt
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml
├── models/             # 训练输出
├── outputs/            # 推理结果
├── weights/            # 预训练模型
├── train.py            # 训练脚本
├── detect.py           # 推理模块
├── utils/
│   ├── balance_aug.py  # 针对稀有类增强
│   └── visualize.py
└── requirements.txt
```

---

### 📦 2. 环境依赖（`requirements.txt`）

```txt
ultralytics==8.2.0
opencv-python==4.8.0
numpy==1.24.3
matplotlib
tqdm
scikit-learn
```

---

### 📄 3. 数据配置文件 `dataset.yaml`

```yaml
# data/dataset.yaml
train: ../data/images/train
val: ../data/images/val
test: ../data/images/test

nc: 8
names: ['bottle', 'grass', 'branch', 'milk-box', 'plastic-bag', 'plastic-garbage', 'ball', 'leaf']
```

---

### 🚀 4. 模型训练（`train.py`）— 重点解决类别不平衡

```python
# train.py
from ultralytics import YOLO

def train_river_litter_detector():
    model = YOLO("yolov8s.pt")  # 或 yolov8m 提升小目标检测

    results = model.train(
        data="data/dataset.yaml",
        imgsz=640,
        epochs=150,
        batch=32,
        device=0,
        name="river_litter_yolov8s",
        project="models",
        patience=50,
        save=True,
        augment=True,
        # 增强策略：提升小目标 & 稀有类
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=15.0,
        translate=0.2,
        scale=0.7,          # 更大缩放范围，覆盖远近目标
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.4,     # ★ 关键：Copy-Paste 增强稀有类（ball, leaf等）
        erasing=0.2,        # 随机擦除，提升鲁棒性
    )

    print(f"✅ 训练完成！最佳模型: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_river_litter_detector()
```

> 💡 **为什么用 `copy_paste`？**  
> 它能将稀有类目标（如 `ball`）从原图裁剪后粘贴到其他图像中，人工增加其出现频率，显著提升召回率。

---

### 🔍 5. 推理检测（`detect.py`）

```python
# detect.py
import cv2
import os
from ultralytics import YOLO

class RiverLitterDetector:
    def __init__(self, model_path="models/river_litter_yolov8s/weights/best.pt"):
        self.model = YOLO(model_path)
        self.class_names = [
            "瓶子", "草", "树枝", "牛奶盒", "塑料袋",
            "塑料垃圾", "球", "树叶"
        ]

    def detect_image(self, image_path, conf=0.2, save_dir="outputs"):
        results = self.model(image_path, conf=conf)
        annotated = results[0].plot()  # 默认英文标签

        # 若需显示中文，可自定义绘图（略，需OpenCV绘制文字）
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, annotated)
        print(f"✅ 检测完成 → {out_path}")

        # 统计各类数量
        counts = {name: 0 for name in self.class_names}
        for cls in results[0].boxes.cls:
            counts[self.class_names[int(cls)]] += 1
        print("📊 检测统计:", counts)
        return counts
```

---

### 🌐 6. 应用场景与部署建议

| 应用领域 | 实现方式 |
|--------|--------|
| **智慧河湖管理** | 无人机定期巡河 + 自动识别垃圾类型与位置 |
| **环保执法** | 岸基摄像头实时监控非法倾倒行为 |
| **水面清洁机器人** | 检测结果引导机械臂精准打捞 |
| **公众参与平台** | 市民上传照片 → AI自动分类 → 积分奖励 |
| **科研分析** | 统计垃圾类型时空分布，溯源污染源 |

**部署方案**：
- **边缘端**：Jetson Nano/Orin + TensorRT 加速
- **云端**：FastAPI 服务 + 前端地图展示（如高德/百度地图叠加告警点）
- **移动端**：导出 ONNX → 集成到 Android/iOS App

---



 **水利局、环保局、智慧城市运营商、河道保洁公司**，实现 **“AI+环保”智能监管闭环**。
