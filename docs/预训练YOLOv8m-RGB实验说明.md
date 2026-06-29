# 预训练 YOLOv8m RGB 实验说明

## 1. 实验目的

当前自研双流模型及四模型融合继续作为稳定基线，最终线上 mAP 为 `0.278432`。新增实验使用开源 COCO 预训练 `yolov8m.pt` 训练 RGB 单模型，先验证预训练与成熟 YOLO 数据增强能否提高测试集泛化，再决定是否加入融合。

该实验不会覆盖现有 E1/E2/E3 权重或最终 JSON，输出统一写入 `outputs/ultralytics/`。

## 2. 新增文件

- `scripts/convert_coco_to_yolo_rgb.py`：将 train/val COCO 标注转换为 YOLO 标签，并整理 RGB train/val/test 图像；
- `scripts/train_ultralytics_rgb.py`：使用预训练 `yolov8m.pt` 训练；
- `scripts/infer_ultralytics_rgb.py`：在 RGB 测试集上推理并导出竞赛 COCO JSON；
- `requirements-ultralytics.txt`：可选 Ultralytics 依赖；
- `tests/test_yolo_conversion.py`：转换坐标、类别映射和数据 YAML 测试。

转换程序默认使用硬链接，不会复制数万张图片的内容。输出应放在原始 `data` 所在文件系统中；AutoDL 环境中的 `data` 已软链接到数据盘，因此默认 `data/yolo_rgb` 合适。

## 3. 安装依赖

在 RTX 5090 服务器的项目根目录执行：

```bash
conda activate dualdet
source /etc/network_turbo
pip install -r requirements-ultralytics.txt
python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__, torch.cuda.get_device_name(0))"
```

不要再次执行主 `requirements.txt`，避免将支持 RTX 5090 的新版 PyTorch 降级为项目早期锁定版本。

## 4. 转换数据

```bash
python scripts/convert_coco_to_yolo_rgb.py \
  --data-root data \
  --output-root data/yolo_rgb \
  --image-mode hardlink
```

预期统计：

| split | 图像数 | 标注框数 |
|---|---:|---:|
| train | 17990 | 316409 |
| val | 1469 | 24490 |
| test | 1000 | 无公开标注 |

类别顺序必须为：`0 car`、`1 truck`、`2 bus`、`3 van`、`4 freight_car`。转换报告保存在 `data/yolo_rgb/conversion_summary.json`，训练配置保存在 `data/yolo_rgb/rgb_dataset.yaml`。

## 5. 冒烟训练

先运行 1 epoch，检查预训练权重下载、显存和验证链路：

```bash
python scripts/train_ultralytics_rgb.py \
  --data data/yolo_rgb/rgb_dataset.yaml \
  --model yolov8m.pt \
  --imgsz 960 \
  --batch 16 \
  --epochs 1 \
  --name yolov8m_rgb_smoke
```

若 CUDA OOM，将 batch 降为 `8`；不需要手动修改学习率，Ultralytics 会依据训练配置处理优化过程。

## 6. 正式训练

```bash
mkdir -p outputs/ultralytics
nohup python scripts/train_ultralytics_rgb.py \
  --data data/yolo_rgb/rgb_dataset.yaml \
  --model yolov8m.pt \
  --imgsz 960 \
  --batch 16 \
  --epochs 100 \
  --patience 20 \
  --close-mosaic 10 \
  --name yolov8m_rgb_pretrained \
  > outputs/ultralytics/yolov8m_rgb_train.log 2>&1 &
echo $! > outputs/ultralytics/yolov8m_rgb_train.pid
```

查看进度：

```bash
tail -f outputs/ultralytics/yolov8m_rgb_train.log
```

最佳权重预期位于：

```text
outputs/ultralytics/yolov8m_rgb_pretrained/weights/best.pt
```

## 7. 测试集导出

```bash
python scripts/infer_ultralytics_rgb.py \
  --checkpoint outputs/ultralytics/yolov8m_rgb_pretrained/weights/best.pt \
  --source data/yolo_rgb/images/test \
  --output outputs/result_yolov8m_rgb_iou070_c001.json \
  --imgsz 960 \
  --batch 16 \
  --conf 0.001 \
  --iou 0.70 \
  --max-det 300
```

脚本会将 YOLO 的 0-4 类别恢复为竞赛要求的 1-5，使用文件名生成 `image_id=1..1000`，并自动校验 JSON。若输出超过网站 5MB 限制，依次将 `--conf` 调为 `0.002`、`0.003` 重新导出，不要修改权重。

统计真实参数量：

```bash
python -c "from ultralytics import YOLO; m=YOLO('outputs/ultralytics/yolov8m_rgb_pretrained/weights/best.pt'); print(sum(p.numel() for p in m.model.parameters())/1e6)"
```

## 8. 决策规则

先单独提交 RGB YOLOv8m：

- 若线上 mAP 明显高于现有最佳单模型 `0.246919`，保留并尝试与当前四模型融合；
- 若达到约 `0.26-0.27` 或更高，优先生成加权融合候选，目标冲击 `0.30+`；
- 若仍低于 `0.25`，暂不训练 TIR YOLOv8m，先检查类别映射、预训练加载和验证/测试域差异。

融合后参数量必须填写所有实际参与推理模型参数量之和，不能继续沿用当前四模型的 `52.91M`。

## 9. TIR 后续实验

RGB YOLOv8m 的验证集 COCO JSON 回放 AP 为 `0.444`，但测试网站 mAP 仅为 `0.090413`，说明纯 RGB 在隐藏测试集上存在严重泛化下降，因此不加入最终融合。下一步使用相同预训练模型和标签训练 TIR 单模型。

先上传更新后的 `dualdet/utils/yolo_conversion.py` 与 `scripts/convert_coco_to_yolo_rgb.py`，然后在服务器转换 TIR：

```bash
python scripts/convert_coco_to_yolo_rgb.py \
  --data-root data \
  --modality tir \
  --output-root data/yolo_tir \
  --image-mode hardlink
```

冒烟训练：

```bash
python scripts/train_ultralytics_rgb.py \
  --data data/yolo_tir/tir_dataset.yaml \
  --model yolov8m.pt \
  --imgsz 960 \
  --batch 16 \
  --epochs 1 \
  --name yolov8m_tir_smoke
```

正式训练：

```bash
mkdir -p outputs/ultralytics
nohup python scripts/train_ultralytics_rgb.py \
  --data data/yolo_tir/tir_dataset.yaml \
  --model yolov8m.pt \
  --imgsz 960 \
  --batch 16 \
  --epochs 100 \
  --patience 20 \
  --close-mosaic 10 \
  --name yolov8m_tir_pretrained \
  > outputs/ultralytics/yolov8m_tir_train.log 2>&1 &
echo $! > outputs/ultralytics/yolov8m_tir_train.pid
```

最佳权重预期位于 `outputs/ultralytics/yolov8m_tir_pretrained/weights/best.pt`。测试导出：

```bash
python scripts/infer_ultralytics_rgb.py \
  --checkpoint outputs/ultralytics/yolov8m_tir_pretrained/weights/best.pt \
  --source data/yolo_tir/images/test \
  --output outputs/result_yolov8m_tir_iou070_c003.json \
  --imgsz 960 \
  --batch 16 \
  --conf 0.003 \
  --iou 0.70 \
  --max-det 300
```

若 JSON 超过 5MB，依次提高 `--conf` 至 `0.005`、`0.006`，并保证输出文件名中的阈值与实际参数一致。先单独提交 TIR 结果；只有线上有效时才与当前四模型融合。
