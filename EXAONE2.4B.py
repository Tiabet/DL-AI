import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_name = "LGAI-EXAONE/EXAONE-Deep-2.4B"
streaming = True    # choose the streaming option

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "역사상 가장 재밌는 영화 후보를 5개 추려줘. 한글로 답변해줘"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

if streaming:
    streamer = TextIteratorStreamer(tokenizer)
    thread = Thread(target=model.generate, kwargs=dict(
        input_ids=input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        streamer=streamer
    ))
    thread.start()

    for text in streamer:
        print(text, end="", flush=True)
else:
    output = model.generate(
        input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=32768,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )
    print(tokenizer.decode(output[0]))
