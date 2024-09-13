import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from model import Transformer  # 确保这个导入与你的模型类文件名匹配

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('/home/yyq/.cache/modelscope/hub/AI-ModelScope/bert-base-uncased')

# 加载预训练的BERT模型
base_model = BertModel.from_pretrained('/home/yyq/.cache/modelscope/hub/AI-ModelScope/bert-base-uncased')

# 定义类别数和输入大小
num_classes = 2  # 二分类
input_size = base_model.config.hidden_size  # 通常是768对于bert-base-uncased

# 创建Transformer实例
model = Transformer(base_model=base_model, num_classes=num_classes, input_size=input_size)

# 加载训练好的权重
model_path = '/data/yyq/Evaluation_Metrics/Sentiment/model_training/model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

class TextDataset(Dataset):
    def __init__(self, text_data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text_data = text_data
        self.max_len = max_len

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, item):
        text = str(self.text_data[item])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }

def load_data(data_path):
    texts = []
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for sentences in data.values():  # Assuming each key corresponds to a list of sentences
                texts.extend(sentences)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {data_path}")
    print(f"Loaded {len(texts)} texts.")
    return texts

# 加载数据
data_path = '/data/yyq/Dataset/Wikipedia/Religious_ideology/sikhism'  # Update the path and file name as per your dataset
texts = load_data(data_path)
dataset = TextDataset(texts, tokenizer, max_len=128)
data_loader = DataLoader(dataset, batch_size=32)

# 推理
results = []
with torch.no_grad():
    for batch in data_loader:
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        predictions = torch.argmax(outputs, dim=1)
        results.extend(predictions.cpu().numpy())

# 分析结果
positive_count = sum(1 for result in results if result == 1)  # Assuming '1' is positive
negative_count = sum(1 for result in results if result == 0)  # Assuming '0' is negative

# 打印统计信息
print(f"Total sentences: {len(results)}")
print(f"Positive sentiment sentences: {positive_count}")
print(f"Negative sentiment sentences: {negative_count}")

# 保存结果到指定文件
#results_path = '/data/yyq/Evaluation_Metrics/Sentiment/Sentiment_Analysis_Imdb-master/Results/Wekipedia/male'  # Update as needed
#with open(results_path, 'w', encoding='utf-8') as file:
    #for result in results:
        #file.write(f'{result}\n')

#print(f'Results saved to {results_path}')
