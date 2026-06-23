# Dual-optical-target-detection
tju计算机视觉实践项目，实现在无人机视角下的双光目标检测任务

## 当前模块

- `dualdet/models/`：双流骨干、QAF 融合、PAN-FPN/P2-PAN 和检测头。
- `configs/`：E1 固定融合、E2 QAF、E3 QAF+P2 的受控模型配置。
- `scripts/analyze_predictions.py`：验证集预测分析工具，输出每类指标、小目标召回和混淆矩阵。
- `docs/成员3-P2小目标检测头与指标分析说明.md`：成员 3 的 P2 Head 与指标分析说明。

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
