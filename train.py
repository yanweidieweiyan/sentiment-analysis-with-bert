import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import os

# --- 1. 配置参数 ---
# 使用字典来集中管理超参数，方便修改和查阅
config = {
    "MODEL_NAME": "distilbert-base-uncased",  # 使用轻量级的DistilBERT
    "BATCH_SIZE": 16,
    "EPOCHS": 3,
    "LEARNING_RATE": 2e-5,
    "MAX_LENGTH": 256, # 截断或填充评论的最大长度
    "OUTPUT_DIR": "./saved_model" # 模型保存路径
}

# --- 2. 数据准备 ---
def load_and_prepare_data(tokenizer):
    """加载IMDb数据集并进行预处理"""
    print("Loading and preparing data...")
    
    # 从Hugging Face Hub加载IMDb数据集
    imdb = load_dataset("imdb")

    # 定义分词函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config["MAX_LENGTH"])

    # 对整个数据集进行分词处理 (map函数会缓存结果，非常高效)
    tokenized_datasets = imdb.map(tokenize_function, batched=True)

    # 移除原始文本列，并将'label'重命名为'labels'，因为模型期望的列名是'labels'
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch") # 将数据集格式化为PyTorch Tensors

    # 从训练集中划分出一小部分作为验证集 (例如10%)
    train_val_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    test_dataset = tokenized_datasets["test"]
    
    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["BATCH_SIZE"])
    val_dataloader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"])

    print("Data preparation complete.")
    return train_dataloader, val_dataloader, test_dataloader

# --- 3. 训练与评估 ---
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, device):
    """训练模型并在每个epoch后进行验证"""
    print(f"Starting training for {config['EPOCHS']} epochs...")
    
    for epoch in range(config["EPOCHS"]):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['EPOCHS']} [Training]")
        
        for batch in train_progress_bar:
            # 将数据移动到指定设备 (GPU or CPU)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清空梯度
            model.zero_grad()
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()

            # 更新进度条信息
            train_progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        # --- 验证阶段 ---
        model.eval()
        total_eval_loss = 0
        total_eval_correct = 0
        
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{config['EPOCHS']} [Validation]")
        for batch in val_progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad(): # 在评估时，不计算梯度以节省内存和计算
                outputs = model(**batch)
            
            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            total_eval_correct += (predictions == batch["labels"]).sum().item()
        
        avg_val_loss = total_eval_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_correct / len(val_dataloader.dataset)
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")
    
    print("Training complete.")
    return model

# --- 4. 最终测试 ---
def test_model(model, test_dataloader, device):
    """在测试集上进行最终评估并打印分类报告"""
    print("\nStarting final evaluation on the test set...")
    model.eval()
    all_preds = []
    all_labels = []

    test_progress_bar = tqdm(test_dataloader, desc="Testing")
    for batch in test_progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    print("\n--- Test Results ---")
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))


# --- 主执行函数 ---
if __name__ == "__main__":
    # 设置设备 (优先使用GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])

    # 加载并准备数据
    train_loader, val_loader, test_loader = load_and_prepare_data(tokenizer)

    # 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(config["MODEL_NAME"], num_labels=2)
    model.to(device) # 将模型移动到设备

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=config["LEARNING_RATE"])
    
    # 训练和评估
    trained_model = train_and_evaluate(model, train_loader, val_loader, optimizer, device)

    # 在测试集上评估最终模型
    test_model(trained_model, test_loader, device)

    # 保存最终模型和分词器
    print(f"\nSaving model to {config['OUTPUT_DIR']}...")
    if not os.path.exists(config['OUTPUT_DIR']):
        os.makedirs(config['OUTPUT_DIR'])
    trained_model.save_pretrained(config['OUTPUT_DIR'])
    tokenizer.save_pretrained(config['OUTPUT_DIR'])
    print("Model saved successfully!")