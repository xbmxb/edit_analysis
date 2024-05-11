import os, json, jsonlines
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
import random
import argparse
import torch
import torch
import torch.nn.functional as F
from tqdm import tqdm
device = 'cuda:4'
tokenizer = LlamaTokenizer.from_pretrained("/data/maxb/tag/model_cache/llama-ckpt/llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("/data/maxb/tag/model_cache/llama-ckpt/llama-2-7b-chat-hf").to(device)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

TEMP = 'Complete the sentence with an entity. Sentence: The capital of Shandong is _ . Completion: Jinan. Sentence: {prompt} _ . Completion: '

def perplexity(model,tok,prompt: str,compl: str, input_token_len, max_input_length: int = None):
    inputs = tok(
        [prompt + compl], return_tensors="pt").to(device)
    # print(prompt, compl, input_token_len, inputs["input_ids"].size(1))
    with torch.no_grad():
        logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]
    target_p = 1
    # print(len(log_probs[input_token_len-1:]))
    for logi in log_probs[input_token_len-1:]:
        target_p *= torch.exp(logi)
    # print(target_p)
    if len(log_probs[input_token_len-1:]) == 1:
        top1_token, top1_text, top1_ppl = generate_score(prompt, model, tokenizer)
        if top1_token.detach().cpu().numpy().tolist()[0] == compl:
            assert top1_ppl == target_p
    if len(log_probs[input_token_len-1:]) < 1 or target_p ==1 :
        return -1
    target_p = target_p.detach().cpu().numpy().tolist()[0]
    return target_p
        
def generate_score(prompt, model, tokenizer):
    batch = tokenizer(prompt, return_tensors='pt', padding=True)
    outputs = model.generate(
        input_ids = batch['input_ids'].to(device),
        attention_mask = batch['attention_mask'].to(device),
        max_new_tokens = 6,
        output_scores = True,
        return_dict_in_generate = True
    )
    
    text = tokenizer.decode(outputs.sequences.detach().cpu().numpy().tolist()[0], skip_special_tokens=True)
    # print(text)
    text = text.split('.')[0]
    # top
    next_token_score = F.softmax(outputs.scores[0], dim=1)
    # print(next_token_score.shape)
    top1_token = torch.argmax(next_token_score, dim=-1)
    top1_text = tokenizer.decode(top1_token.detach().cpu().numpy().tolist()[0], skip_special_tokens=True)
    top1_ppl = next_token_score[0,top1_token.detach().cpu().numpy().tolist()[0]].detach().cpu().numpy()
    # print(top1_token, top1_text, top1_ppl)
    return top1_token, top1_text, top1_ppl

def get_prob(request):
    # prompt = TEMP.replace('{prompt}', request['prompt'].format(request['subject']))
    # prompt = 'Singled Out debuted on'
    prompt = request['prompt'].format(request['subject'])
    batch = tokenizer(prompt, return_tensors='pt', padding=True)
    input_token_len = batch['input_ids'].size(1)
    outputs = model.generate(
        input_ids = batch['input_ids'].to(device),
        attention_mask = batch['attention_mask'].to(device),
        max_new_tokens = 6,
        output_scores = True,
        return_dict_in_generate = True
    )
    
    text = tokenizer.decode(outputs.sequences.detach().cpu().numpy().tolist()[0], skip_special_tokens=True)
    # print(text)
    text = text.split('.')[0]
    top1_token, top1_text, top1_ppl = generate_score(prompt, model, tokenizer)
    top1_ppl = top1_ppl.tolist()
    # generate, target true / new
    target_true = request['target_true']["str"]
    target_new = request['target_new']["str"]
    generation = text.replace(prompt, '')
    if generation.startswith(target_true):
        generation = target_true
    elif generation.startswith(target_new):
        generation = target_new
    
    target_p_generation = perplexity(model, tokenizer, prompt, generation, input_token_len)
    target_p_true = perplexity(model, tokenizer, prompt, ' ' + target_true, input_token_len)
    target_p_new = perplexity(model, tokenizer, prompt, ' ' + target_new, input_token_len)
    ret_dic = {
        'top1_text':  top1_text,
        'top1_ppl': top1_ppl,
        'generation': text,
        'target_p_generation': target_p_generation,
        'target_p_true': target_p_true,
        'target_p_new': target_p_new
    }
    return ret_dic

with open('/data/maxb/mememe/EasyEdit/cf/counterfact.json', 'r') as f:
    data = json.load(f)

output = []
for i in tqdm(range(len(data))):
    ret_dic = get_prob(data[i]['requested_rewrite'])
    ret_dic['case_id'] = data[i]['case_id']
    # print(ret_dic)
    output.append(ret_dic)

with open('see_logits_v2.json', 'w') as f:
    json.dump(output, f, indent=2)
