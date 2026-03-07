import runpod
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen3-TTS"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

def handler(event):
    text = event["input"]["text"]

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs)

    return {"result": output.tolist()}

runpod.serverless.start({"handler": handler})
