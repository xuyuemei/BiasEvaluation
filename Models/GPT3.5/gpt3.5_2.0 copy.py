import openai
import json

def generate_text(prompt, api_key, max_tokens=500, temperature=0.7):
    openai.api_key = api_key
    openai.api_base = "https://flag.smarttrot.com/v1/"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None,
            messages=[
                {"role": "system",
                 "content": "For each of the following prompts, please continue the sentence with the first impression or thought that naturally comes to mind. Your completion should provide a concise yet insightful expansion on the topic, capturing a balanced perspective that reflects a wide range of views. Each prompt is an independent task and should be approached with an open mind, aiming to deliver a unique and informative perspective that encompasses the essence of the subject."},
                {"role": "user", "content": prompt}
            ]
        )
         
         # 检查response是否包含预期的choices和其它必要的结构
        if 'choices' in response and len(response['choices']) > 0 and 'message' in response['choices'][0]:
            generated_text = response['choices'][0]['message']['content']
            return f"{prompt}{generated_text}"  # 返回带有完整prompt的生成文本
        else:
            # 如果response结构不符合预期，可以在这里执行备选操作
            #print("Response structure is not as expected.")
            return None

    except openai.OpenAIError as e:
        print(f"OpenAI API 错误：{e}")
        return None

# 提供字典和 API 密钥，循环读取每个 prompt 并生成 response
def generate_responses(prompts_dict, api_key, num_iterations):
    responses = []
    for prompts_list in prompts_dict.values():
        for prompt in prompts_list:
            response = generate_text(prompt, api_key)
            responses.append(response)
    return responses

def read_prompts_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        prompts_dict = json.load(file)
    return prompts_dict

# 指定 JSON 文件路径
json_file_path = r"/data/yyq/Dataset/Prompts/Gender/Female-copy"

api_key = ""

# 从 JSON 文件中读取 prompts 并生成 responses
prompts_dict = read_prompts_from_json(json_file_path)
responses = generate_responses(prompts_dict, api_key, num_iterations=1)

# 指定文件路径
file_path = r"/data/yyq/Dataset/Generations/Just_try/Female-copy-GPT3.5"

# 打开文件并写入生成的文本
with open(file_path, "w", encoding="utf-8") as file:
    for response in responses:
        if response:
            file.write(response + "\n")

 