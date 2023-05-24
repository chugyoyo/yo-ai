import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchviz import make_dot

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)

# model file: debug at PreTrainedModel.from_pretrained#config
# print model object
print(model)

# model graph
input_ids = torch.tensor([[[1, 2, 3, 4, 5]]])
output = model(input_ids)[0]
print(output)
loss = output.sum()
loss.backward()
graph = make_dot(output, params=dict(model.named_parameters()))
# add dot to path
graph.render("model_graph")

# result output
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
input_text = "translate the word 'billion' to Chinese"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
