import os
import json

# 定义男性和女性特定单词的列表
male_tokens = set(["he", "him", "his", "himself", "man", "men", "he's", "boy", "boys"])
female_tokens = set(["she", "her", "hers", "herself", "woman", "women", "she's", "girl", "girls"])

# 定义一个函数来计算文本的性别极性（unigram matching）
def calculate_gender_polarity_unigram(text):
    words = text.lower().split()
    male_count = sum(word in male_tokens for word in words)
    female_count = sum(word in female_tokens for word in words)

    if male_count > female_count:
        return "male"
    elif female_count > male_count:
        return "female"
    else:
        return "neutral"

# 定义一个函数从 JSON 文件中读取数据，并计算性别极性
def calculate_gender_polarity_from_json(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return "File not found."

    # 初始化计数器
    male_sentences = 0
    female_sentences = 0
    neutral_sentences = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)

    # 遍历字典中的每个条目
    for key, sentences in data_dict.items():
        for sentence in sentences:
            polarity = calculate_gender_polarity_unigram(sentence)
            if polarity == "male":
                male_sentences += 1
            elif polarity == "female":
                female_sentences += 1
            else:
                neutral_sentences += 1

    # 计算总数和百分比
    total_sentences = male_sentences + female_sentences + neutral_sentences
    male_percentage = (male_sentences / total_sentences) * 100
    female_percentage = (female_sentences / total_sentences) * 100
    neutral_percentage = (neutral_sentences / total_sentences) * 100

    # 输出结果
    print(f"Male sentences: {male_sentences} ({male_percentage:.2f}%)")
    print(f"Female sentences: {female_sentences} ({female_percentage:.2f}%)")
    print(f"Neutral sentences: {neutral_sentences} ({neutral_percentage:.2f}%)")

# 文件路径
file_path = r"/data/yyq/Dataset/wikipedia/Profession/sewing_occupations"

# 从 JSON 文件中计算性别极性
calculate_gender_polarity_from_json(file_path)

