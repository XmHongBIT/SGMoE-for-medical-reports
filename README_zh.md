# SGMoE：完整模型的通用实现

这版只公开完整 SGMoE，不包含特定数据集预设、基线、消融分支或模型模式选择。
类别从用户的训练文件中读取，专家数量和分组由语义聚类流程生成。完整模型始终包含：

- LoRA 共享编码器与独立训练后冻结的语义编码器。
- 语义中心先验与学习门控的固定融合。
- 类别分组专家及最终类别路由。
- 分类损失、门控损失与动态专家分类损失。

输入采用 CSV 或 JSONL，每条记录包含文本和一个类别标签，字段名可以配置。
不要求文件名携带标签，也不固定类别编号、类别数量或专家数量。公开包不含任何数据文件。

复制 `configs/example.json` 为 `configs/local.json`，填写自己的数据与编码器路径。
相对路径以配置文件位置为基准；编码器也可使用 Hugging Face 模型 ID。
模型维度自动读取，LoRA 目标模块按编码器结构配置。

```bash
python -m pip install -r requirements.txt
python prepare.py --config configs/local.json --device cuda:0
python train.py --config configs/local.json --dry-run
python train.py --config configs/local.json --device cuda:0
python predict.py --checkpoint outputs/best.pt --input path/to/unlabeled.csv --output path/to/predictions.csv --device cuda:0
```

`prepare.py` 训练语义编码器并仅从训练文本构建中心和分组，验证集用于选取编码器。
`train.py` 运行完整模型，按验证集 macro F1 选择最佳模型，保存 macro/weighted F1。
`predict.py` 接收无标签文本，输出预测类别和概率。测试文件可不配置。

两个训练入口均支持 `--resume`；已完成结果核验后跳过，缺失或损坏断点会停止并保留文件。
恢复模型、优化器、调度器和随机状态，当前未完成的 epoch 从头继续。
语义文件缺失或不匹配会报错，不会自动降为无先验模型。

完整配置、数据接口和数学说明见 [英文 README](README.md)、[数据格式](docs/DATA.md)、[模型说明](docs/MODEL.md)。
MIT 许可证适用于代码。真实数据与权重不在本仓库分发，通用实现不等于原实验成绩的自动复现。
