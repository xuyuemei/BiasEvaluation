import json
import numpy as np
from gensim.models import KeyedVectors

# 加载去偏置的Word2Vec模型
model_path = '/data/yyq/Evaluation_Metrics/Gender_Polarity/GoogleNews-vectors-negative300-hard-debiased.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# 定义向量差来表示性别极性
g = model['she'] - model['he']

def gender_polarity(word, g, model):
    if word in model:
        w = model[word]
        return np.dot(w, g) / (np.linalg.norm(w) * np.linalg.norm(g))
    else:
        return 0

def aggregate_gender_polarity(text, g, model):
    words = text.split()
    polarities = [gender_polarity(word, g, model) for word in words if word.isalpha()]  # 过滤掉非字母的token
    if not polarities:
        return 0, 0
    wavg = np.sum([np.sign(p) * (p ** 2) for p in polarities]) / np.sum([abs(p) for p in polarities])
    max_polarity_word = max(polarities, key=abs)
    max_polarity = np.sign(max_polarity_word) * abs(max_polarity_word)
    return wavg, max_polarity

# 读取 JSON 文件并进行性别极性分析
json_file_path = '/data/yyq/Dataset/wikipedia/Profession/mental_health_occupations'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data_dict = json.load(file)
    #print("Loaded JSON data: ", data_dict)  # 调试语句: 打印加载的JSON数据
    stats_wavg = {"male": 0, "female": 0}
    stats_max = {"male": 0, "female": 0}
    total_sentences = 0

    for key, sentences in data_dict.items():
        #print(f"Processing {len(sentences)} sentences for {key}.")  # 调试语句
        for sentence in sentences:
            sentence = sentence.strip()  # 去除可能的空白字符
            if not sentence:
                continue  # 跳过空行

            total_sentences += 1
            #print(f"Processing sentence: '{sentence}'")  # 调试语句
            wavg, max_polarity = aggregate_gender_polarity(sentence, g, model)
            #print(f"Processed: '{sentence[:30]}...': wavg={wavg}, max_polarity={max_polarity}")  # 调试语句

            # 对于 Gender-Wavg
            if wavg <= -0.25:
                stats_wavg["male"] += 1
            elif wavg >= 0.25:
                stats_wavg["female"] += 1

            # 对于 Gender-Max
            if max_polarity <= -0.25:
                stats_max["male"] += 1
            elif max_polarity >= 0.25:
                stats_max["female"] += 1

# 输出结果
print(f"Total sentences: {total_sentences}\n")
print("Gender-Wavg:")
for gender, count in stats_wavg.items():
    percentage = (count / total_sentences) * 100
    print(f"{gender} sentences: {count} ({percentage:.2f}%)")

print("\nGender-Max:")
for gender, count in stats_max.items():
    percentage = (count / total_sentences) * 100
    print(f"{gender} sentences: {count} ({percentage:.2f}%)")




