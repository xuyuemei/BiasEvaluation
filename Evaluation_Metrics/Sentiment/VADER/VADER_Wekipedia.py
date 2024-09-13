# 导入所需的库和模块
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 指定本地vader_lexicon文件路径
lexicon_file = "/data/yyq/Evaluation_Metrics/Sentiment/VADER/vader_lexicon.txt"

# 初始化情感强度分析器，并指定本地lexicon文件
sid = SentimentIntensityAnalyzer(lexicon_file=lexicon_file)

# 文件路径（请根据实际路径修改）
file_path = r"/data/yyq/Dataset/Wikipedia/Religious_ideology/sikhism"  # 替换为你的文件名

# 统计变量
positive_count = 0
negative_count = 0
total_count = 0

# 打开并读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取并解析JSON数据
    data = json.load(file)

# 对字典中的每个人物描述进行情感分析
for key, descriptions in data.items():
    for description in descriptions:
        # 判断文本非空
        if description:
            total_count += 1
            # 使用VADER进行情感分析
            scores = sid.polarity_scores(description)
            # 根据复合分数判断情感极性，并更新统计
            if scores['compound'] > 0.5:
                positive_count += 1
            elif scores['compound'] < -0.5:
                negative_count += 1
            # 可以选择打印每行的分析结果
            #print(f"Actor: {key}\nDescription: {description}\nScores: {scores}\n")

# 计算并打印正面和负面句子的数量及百分比
print(f"Total descriptions analyzed: {total_count}")
print(f"Positive descriptions: {positive_count} ({positive_count/total_count*100:.2f}%)")
print(f"Negative descriptions: {negative_count} ({negative_count/total_count*100:.2f}%)")
