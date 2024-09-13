import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# 指定本地模型路径
model_dir = "/data/yyq/Models/T5/T5-base-model"

# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# 准备输入文本，从JSON文件读取
prompt_file_path = "Dataset/Prompts/Profession/engineering_branches"
with open(prompt_file_path, 'r', encoding='utf-8') as file:
    prompts = json.load(file)

# 生成文本，并保存到新文件
output_file_path = "/data/yyq/Dataset/Generations/T5/Profession/engineering_branches"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for name, texts in prompts.items():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                max_length=500,
                temperature=0.9,
                num_return_sequences=1
            )
            completed_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            output_file.write(f"{name}: {completed_text}\n")

print("文本生成完成并已保存。")
