# 情感分析项目：基于Hugging Face Transformers的IMDb电影评论分析

本项目是机器学习课程的结课作业，旨在使用预训练的Transformer模型（DistilBERT）对经典的IMDb电影评论数据集进行情感分析，并探索“预训练-微调”这一强大的迁移学习范式。

## 项目亮点
- **模型**: 使用轻量级且高效的 `distilbert-base-uncased` 模型。
- **框架**: 基于 PyTorch 和 Hugging Face 生态（`transformers`, `datasets`）。
- **设备**: 成功配置并使用 CUDA (NVIDIA GPU) 进行加速训练。
- **性能**: 在测试集上达到了约 93% 的准确率。

## 如何运行

1.  **克隆本仓库**:
    ```bash
    git clone https://github.com/<你的用户名>/sentiment-analysis-with-bert.git
    cd sentiment-analysis-with-bert
    ```

2.  **创建并激活虚拟环境**:
    *(建议使用 Python 3.9+)*
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    *(如果遇到网络问题，可以尝试使用国内镜像源: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`)*

4.  **运行训练脚本**:
    ```bash
    python train.py
    ```
    脚本将自动下载数据集和模型，进行训练、评估，并最终将训练好的模型保存在 `saved_model` 文件夹下（该文件夹已被`.gitignore`忽略）。

## 最终实验结果

模型在IMDb测试集（25,000条评论）上评估的最终性能如下：--- Evaluation Results ---
              precision    recall  f1-score   support

    Negative       0.93      0.89      0.91     12500
    Positive       0.89      0.93      0.91     12500

    accuracy                           0.91     25000
    macro avg       0.91      0.91      0.91     25000
    weighted avg       0.91      0.91      0.91     25000

## 项目总结与心得
通过这个项目，我不仅学习了如何应用Transformer模型解决实际问题，更重要的是，我亲手解决了从环境配置、依赖冲突到GPU驱动等一系列工程难题。这个过程让我深刻理解到，一个成功的机器学习项目不仅需要扎实的理论知识，更需要强大的动手实践和问题解决能力。

## GitHub地址
[https://github.com/yanweidieweiyan/sentiment-analysis-with-bert](https://github.com/yanweidieweiyan/sentiment-analysis-with-bert)
