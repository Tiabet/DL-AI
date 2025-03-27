import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_name = "LGAI-EXAONE/EXAONE-Deep-7.8B"
streaming = True    # choose the streaming option

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    load_in_8bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "허깅페이스를 사용해서 작은 LLM을 훈련하는 코드를 작성해줘."}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

if streaming:
    streamer = TextIteratorStreamer(tokenizer)
    # thread = Thread(target=model.generate, kwargs=dict(
    #     input_ids=input_ids.to("cuda"),
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_new_tokens=32768,
    #     do_sample=True,
    #     temperature=0.6,
    #     top_p=0.95,
    #     streamer=streamer
    # ))
    thread = Thread(target=model.generate, kwargs=dict(
        input_ids=input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=32768,  # 토큰 수를 256으로 제한
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        early_stopping=True,  # 필요 시 추가
        streamer=streamer
    ))

    thread.start()

    for text in streamer:
        print(text, end="", flush=True)
else:
    # output = model.generate(
    #     input_ids.to("cuda"),
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_new_tokens=32768,
    #     do_sample=True,
    #     temperature=0.6,
    #     top_p=0.95,
    # )
    output = model.generate(
        input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=32768,  # 토큰 수를 256으로 제한
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        early_stopping=True  # 필요 시 추가
    )

    print(tokenizer.decode(output[0]))
