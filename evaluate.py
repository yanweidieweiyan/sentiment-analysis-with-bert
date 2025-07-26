import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm

# --- 配置 ---
MODEL_PATH = "./saved_model"  # 指定你保存好的模型的路径
BATCH_SIZE = 16 # 可以适当调大，例如32，评估时内存占用较小
MAX_LENGTH = 256

# --- 加载数据和模型 ---
print(f"Loading tokenizer and model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# --- 准备测试数据 ---
print("Loading and preparing test data...")
imdb = load_dataset("imdb", split="test") # 只需要加载测试集

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

test_dataset = imdb.map(tokenize_function, batched=True)
test_dataset = test_dataset.remove_columns(["text"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch")

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("Test data ready.")

# --- 执行评估 ---
print("\nStarting evaluation...")
model.eval()
all_preds = []
all_labels = []

progress_bar = tqdm(test_dataloader, desc="Evaluating")
for batch in progress_bar:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    all_preds.extend(predictions.cpu().numpy())
    all_labels.extend(batch["labels"].cpu().numpy())

# --- 打印结果 ---
print("\n--- Evaluation Results ---")
print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))