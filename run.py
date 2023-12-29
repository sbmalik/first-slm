import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-125m"
m_model_id = "sbmalik/finetuning-slm"

model = AutoModelForCausalLM.from_pretrained(model_id)
config = PeftConfig.from_pretrained(m_model_id)
model = PeftModel.from_pretrained(model, m_model_id)
tokenizer = AutoTokenizer.from_pretrained(m_model_id)

# Get raw logits
# inputs = tokenizer(text, return_tensors="pt")
# res = model(**inputs).logits

parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=20)
args = parser.parse_args()
max_length = args.max_length


print()
print("*" * 20)
print("Enter a text and press enter to get the model's response.")
print("Press Ctrl+C to exit.")
print("*" * 20)
print()

while True:
    try:
        text = input("User: ")
    except KeyboardInterrupt:
        print()
        break
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(input_ids=input_ids, max_length=max_length)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Model: {output_text}")
    print("-" * 20)
