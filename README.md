# 股票/黄金价格预测系统

> 基于LSTM和贝叶斯优化的股票/黄金价格预测系统，提供未来价格走势预测和趋势分析。支持多种金融资产预测，包括股票和黄金期货，帮助投资者做出更明智的决策。

基于深度学习和贝叶斯优化的时间序列预测工具，用于预测股票和黄金期货价格走势。本系统使用LSTM神经网络模型，结合技术指标分析，并通过贝叶斯优化自动寻找最佳超参数配置。


## 主要功能

- **数据获取**：通过Yahoo Finance API获取历史价格数据
- **技术分析**：自动计算RSI、MACD等技术指标作为特征
- **贝叶斯优化**：使用scikit-optimize寻找最优模型超参数
- **LSTM模型**：使用TensorFlow构建长短时记忆神经网络
- **价格预测**：支持单日预测和日期范围预测
- **趋势分析**：提供涨跌趋势预测和置信度评估

## 安装指南

1. 克隆仓库：
```bash
git clone https://github.com/OpenSeeker/stock-prediction.git
cd stock-prediction
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate    # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

运行主程序：
```bash
python Predict.py
```

### 菜单选项：
1. **训练最终模型**：使用优化后的超参数训练模型
2. **预测下一交易日**：预测明日价格走势
3. **预测指定日期范围**：预测多日价格走势（实验性）
4. **运行贝叶斯优化**：自动寻找最佳模型参数
5. **退出程序**

### 示例：
```python
请输入选项 (1-5): 2
请输入要预测的股票/黄金代码 (例如 'AAPL', 'GC=F'): GC=F

--- GC=F 在 2025-08-14 的预测 (目标: 百分比变化) ---
预测百分比变化: 0.85%
预测收盘价: 1980.50
目标日期前一交易日 (2025-08-13) 收盘价: 1963.80
预测趋势: 上涨 ▲
```

## 文件结构

```
stock-prediction/
├── Predict.py             # 主程序
├── README.md              # 项目文档
├── LICENSE                # 开源许可证
├── requirements.txt       # 依赖列表
├── saved_models/          # 训练好的模型
├── saved_scalers/         # 特征缩放器
├── saved_columns/         # 特征列名
└── saved_configs/         # 优化配置
```

## 技术细节

- **模型架构**：双层LSTM + Dropout + Dense层
- **正则化**：L1/L2正则化防止过拟合
- **特征工程**：RSI、MACD、布林带等技术指标
- **超参数优化**：高斯过程优化（Bayesian Optimization）
- **评估指标**：方向准确率（预测涨跌正确率）

## 贡献指南

欢迎贡献代码！请遵循以下步骤：
1. Fork本仓库
2. 创建新分支（`git checkout -b feature/your-feature`）
3. 提交修改（`git commit -am 'Add some feature'`）
4. 推送分支（`git push origin feature/your-feature`）
5. 创建Pull Request

## 许可证

本项目采用 [MIT 许可证](LICENSE) - 详情请查看LICENSE文件。

