from transformers import GPT2LMHeadModel, GPT2Tokenizer


'''
GPT2-Small: The GPT2-Small model has approximately 124 million parameters.
 The storage size of the GPT2-Small model is approximately 0.5 GB.

GPT2-Medium: The GPT2-Medium model has approximately 345 million parameters.
 The storage size of the GPT2-Medium model is approximately 1.5 GB.

GPT2-Large: The GPT2-Large model has approximately 774 million parameters.
 The storage size of the GPT2-Large model is approximately 3 GB.

GPT2-XL: The GPT2-XL model has approximately 1.5 billion parameters. 
The storage size of the GPT2-XL model is approximately 6 GB.
'''
# 加载预训练的GPT-2模型和分词器
# model_name = 'gpt2'  # 使用预训练的GPT-2模型
model_name = 'GPT2-XL'  # 使用预训练的GPT-2模型
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 输入文本
input_text = "translate the word 'billion' to Chinese"

# 对输入文本进行分词
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 使用模型生成文本
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 输出生成的文本
print(generated_text)
