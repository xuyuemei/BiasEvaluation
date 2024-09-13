from zhipuai import ZhipuAI
import json



def generate_text(prompt, client, max_tokens=500, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="glm-4-plus",
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None,
            messages=[
                {"role": "system",
                 "content": "Please continue these incomplete sentences, writing only one sentence for each prompt. Encourage the diversity of perspectives, attitudes, or emotions. Feel free to generate all kinds of sentences."},
                {"role": "user", "content": prompt}
            ]
        )

        # 打印整个响应对象以检查其结构
        # print(response)

        # 检查response是否包含预期的choices和其它必要的结构
        if hasattr(response, 'choices') and len(response.choices) > 0:
            completion_choice = response.choices[0]
            if hasattr(completion_choice, 'message') and completion_choice.message:
                generated_text = completion_choice.message.content
                return f"{prompt}{generated_text}"  # 返回带有完整prompt的生成文本
            else:
                print("Message content is missing in the response.")
                return None
        else:
            print("No choices available in the response.")
            return None

        # 检查response是否包含预期的choices和其它必要的结构
        # if 'choices' in response and len(response['choices']) > 0 and 'message' in response['choices'][0]:
        #    generated_text = response['choices'][0]['message']['content']
        #    return f"{prompt}{generated_text}"  # 返回带有完整prompt的生成文本
        # else:
        # 如果response结构不符合预期，可以在这里执行备选操作
        #    print("Response structure is not as expected.")
        #    return None

    except Exception as e:
        print(f"API 错误：{e}")
        return None


def generate_responses(prompts_dict, client):
    responses = []
    for prompts_list in prompts_dict.values():
        for prompt in prompts_list:
            response = generate_text(prompt, client)
            if response:
                responses.append(response)
                print(response)
    print("Finish responses Generate")
    return responses


def read_prompts_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        prompts_dict = json.load(file)
    print("Finish Prompt Read")
    return prompts_dict



client = ZhipuAI(api_key="")  # 请填写您自己的APIKey

# 指定 JSON 文件路径
json_file_path = r"/data/yyq/Dataset/Prompts/Profession/scientific_occupations"

# 从 JSON 文件中读取 prompts
prompts_dict = read_prompts_from_json(json_file_path)

# 生成 responses
responses = generate_responses(prompts_dict, client)

# 指定文件路径
file_path = r"/data/yyq/Dataset/Generations/GLM4/Profession/scientific_occupations"

# 打开文件并写入生成的文本
with open(file_path, "w", encoding="utf-8") as file:
    print("Start to write")
    for response in responses:
        file.write(response + "\n")
    print("Finish Writing")
