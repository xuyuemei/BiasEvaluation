import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# 设定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 指定本地模型路径
model_dir = "/data/yyq/Models/T5/T5-base-model"

# 加载模型和分词器
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)

# 从JSON文件读取prompts
prompt_file_path = "/data/yyq/Dataset/Prompts/Gender/Female-copy"
with open(prompt_file_path, 'r', encoding='utf-8') as file:
    prompts = json.load(file)

# 生成文本，并保存到新文件
output_file_path = "/data/yyq/Dataset/Generations/Just_try/Female-copy-T5"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for name, texts in prompts.items():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                max_length=500,
                temperature=0.9,
                num_return_sequences=1
            )
            completed_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            output_file.write(f"{name}: {completed_text}\n")

print("文本生成完成并已保存。")
