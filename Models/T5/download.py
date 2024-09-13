from transformers import T5Tokenizer, T5ForConditionalGeneration

# 从Hugging Face直接下载
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 将模型和分词器保存到本地目录
save_directory = "/data/yyq/Models/T5"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)