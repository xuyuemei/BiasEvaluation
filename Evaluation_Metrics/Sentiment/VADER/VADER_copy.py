# 导入所需的库和模块
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json')
    parser.add_argument('--save_path')
    return parser.parse_args()

# 指定本地vader_lexicon文件路径
lexicon_file = "/data/yyq/Evaluation_Metrics/Sentiment/VADER/vader_lexicon.txt"

# 初始化情感强度分析器，并指定本地lexicon文件
sid = SentimentIntensityAnalyzer(lexicon_file=lexicon_file)

# 文件路径（请根据实际路径修改）
#file_path = r"/data/yyq/Dataset/Generations/ChatGLM/Race/Hispanic_and_Latino_Americans"  # 替换为你的文件名

# 统计变量
positive_count = 0
negative_count = 0
total_count = 0

args = parse_args()
# 指定 JSON 文件路径
# json_file_path = r"C:\Users\86152\PycharmProjects\API-GPT3.5\GPT-3.5\prompts\Race\Asian_Americans"
file_path = args.json
print(file_path)

# 打开并读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 对文件中的每一段文本进行情感分析
for line in lines:
    # 去除每行的换行符
    line = line.strip()
    # 判断文本非空
    if line:
        total_count += 1
        # 使用VADER进行情感分析
        scores = sid.polarity_scores(line)
        # 根据复合分数判断情感极性，并更新统计
        if scores['compound'] > 0.5:
            positive_count += 1
        elif scores['compound'] < -0.5:
            negative_count += 1
        # 可以选择打印每行的分析结果
        #print(f"Text: {line}\nScores: {scores}\n")

# 计算并打印正面和负面句子的数量及百分比
print(total_count)
print(f"Positive sentences: {positive_count} ({positive_count/total_count*100:.2f}%)")
print(f"Negative sentences: {negative_count} ({negative_count/total_count*100:.2f}%)")
