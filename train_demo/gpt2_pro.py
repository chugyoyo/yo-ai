import torch
from torch import nn, optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 准备数据集
input_text = ["i have a "]
labels = [0]  # 样本标签

# 加载预训练模型
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 添加 padding token
model = GPT2LMHeadModel.from_pretrained(model_name)

# 调整模型架构
model.resize_token_embeddings(len(tokenizer))
model.fc = nn.Linear(model.config.hidden_size, 2)  # 示例中假设有2个类别

# 准备数据
encoded_inputs = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
labels = torch.tensor(labels)

# 设置优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 微调训练
num_epochs = 10

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    loss = criterion(logits.view(-1, 2), labels.view(-1))  # 2表示两个类别
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 评估模型性能
# ...

# 应用和推理
# ...
