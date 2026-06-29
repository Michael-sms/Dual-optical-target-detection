# Dual-optical-target-detection
tju计算机视觉实践项目，实现在无人机视角下的双光目标检测任务

## 当前模块

- `dualdet/models/`：双流骨干、QAF 融合、PAN-FPN/P2-PAN 和检测头。
- `configs/`：E1 固定融合（v8s/v8m）、E2 QAF、E3 QAF+P2 的受控模型配置。
- `scripts/analyze_predictions.py`：验证集预测分析工具，输出每类指标、小目标召回和混淆矩阵。
- `scripts/convert_coco_to_yolo_rgb.py`、`train_ultralytics_rgb.py`、`infer_ultralytics_rgb.py`：开源 COCO 预训练 YOLOv8m RGB/TIR 单模态扩展实验链路。
- `docs/成员3-P2小目标检测头与指标分析说明.md`：成员 3 的 P2 Head 与指标分析说明。

预训练 RGB/TIR 扩展实验的完整命令见 `docs/预训练YOLOv8m-RGB实验说明.md`。训练权重独立写入 `outputs/ultralytics/`。

## 最终提交结果

最终采用“预训练 TIR YOLOv8m + 原四模型”的五模型结果级融合：

- 单模型线上最佳：预训练 TIR YOLOv8m，mAP `0.327517`；
- 最终 JSON：`outputs/result_final.json`；
- 融合组成：TIR YOLOv8m + E1 fixed fusion v8s + E1 fixed fusion v8m + E2 QAF + E3 QAF+P2；
- 融合策略：TIR 权重 `1.0`、阈值 `0.002`；原四模型权重 `0.25`、阈值 `0.05`；按类别执行 `NMS IoU=0.68`；
- 模型大小：`78.75M`；
- 最终线上 mAP：`0.357833`。

## 成员 3 快速命令

生成验证集分析表：

```powershell
python scripts/analyze_predictions.py `
  --annotations data/val/val.json `
  --predictions outputs/e3_val_predictions.json `
  --markdown-output outputs/analysis/e3_analysis.md `
  --json-output outputs/analysis/e3_analysis.json
```

正式 E1-E3 数值对比需要等待训练 checkpoint 和验证集预测结果。

其他成员现在都可以正式开展其实现工作，但部分最终产出仍依赖训练结果:
成员2:已完全解除模型接口阻塞，可以加载E1/E2配置，完成数据管线、损失、训练循环并正式训练
成员3:可以完成指标分析工具和P2Head;正式E1-E3对比要等待成员2的checkpoint
成员4:可以完成解码、NMS、评估、JSON导出、README和报告框架;最终结果文件和报告数据要等待最终checkpoint
