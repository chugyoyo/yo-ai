import openai

# 设置OpenAI API密钥
openai.api_key = 'YOUR_API_KEY'

# 定义生成文本的函数
def generate_text(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n = 1,
        stop=None,
        temperature=0.7
    )
    text = response.choices[0].text.strip()
    return text

# 提供一个提示并生成文本
prompt = "Once upon a time"
generated_text = generate_text(prompt)

# 输出生成的文本
print(generated_text)
