from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B")
model = AutoModelWithLMHead.from_pretrained("cerebras/Cerebras-GPT-13B")

text = "What are you capable of doing?"
input_ids = tokenizer.encode(text, return_tensors="pt")
output = model.generate(input_ids)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)