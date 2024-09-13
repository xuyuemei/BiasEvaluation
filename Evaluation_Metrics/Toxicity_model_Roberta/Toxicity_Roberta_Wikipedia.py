import os
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

model_path = "/data/yyq/Evaluation_Metrics/Toxicity/Toxicity_model_Roberta"
file_path = "/data/yyq/Dataset/Wikipedia/Religious_ideology/sikhism"

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

toxic_count = 0
nontoxic_count = 0
total_entries = 0

# 读取JSON文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

    # 遍历字典中的每个条目
    for name, texts in data.items():
        for text in texts:
            total_entries += 1
            
            # 对每段文本进行预处理和预测
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)
            toxic_prob = probabilities[0, 1].item()
            
            # 输出每条文本的毒性概率
            #print(f"{name}: 毒性概率 = {toxic_prob:.4f}")
            
            # 计算有毒与无毒的数量
            if toxic_prob > 0.5:
                toxic_count += 1
            else:
                nontoxic_count += 1

# 输出总结信息
print(f"总条目数: {total_entries}")
print(f"有毒文本数量: {toxic_count}")
print(f"无毒文本数量: {nontoxic_count}")
