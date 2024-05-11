from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams, ROMEHyperParams, IKEHyperParams
import os, json, jsonlines
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
import random
import argparse
from easyeditor import apply_ike_to_model
from sentence_transformers import SentenceTransformer
from easyeditor.models.ike.util import encode_ike_facts
from tqdm import tqdm
import torch, copy, gc

data_path = {
    # 'counterfact': '/data/maxb/mememe/EasyEdit/cf/counterfact.json',
    # '100cfapi': '/data/maxb/mememe/EasyEdit/cf/tryapi100_full.json',
    # '100cfapi70': '/data/maxb/mememe/EasyEdit/cf/tryapi100_full_70b.json',
    # 'biov0': '/data/maxb/mememe/EasyEdit/cf/bio.json',
    # 'npcv0': '/data/maxb/mememe/EasyEdit/cf/npc.json',
    # '100cfwiki30b': '/data/maxb/mememe/EasyEdit/cf/tryapi100_wiki_full_33b_pronoun.json',
    # '100cfwiki30b_notargettrue': '/data/maxb/mememe/EasyEdit/cf/tryapi100_wiki_full_33b_pronoun_notargettrue.json',
    # '100cfdial': '/data/maxb/mememe/EasyEdit/cf/try100_betterdialog.json',
    # '100cfanother': '/data/maxb/mememe/EasyEdit/cf/tryapi100_wiki_another_full_33b_pronoun_notargettrue.json',
    # '100full1227': '/data/maxb/mememe/EasyEdit/cf/tryapi100_1227full.json',
    # '100diainsert': '/data/maxb/mememe/EasyEdit/cf/try100_dialog_insert.json',
    # 'counterfactbucket0': '/data/maxb/mememe/EasyEdit/cf/cf_bucket/counterfact_bucketed0.json',
    # 'counterfactbucket1': '/data/maxb/mememe/EasyEdit/cf/cf_bucket/counterfact_bucketed1.json'
    '2kfull': './cf/0111_2k_cf/try2k_0111full.json',
    '2kfull0114': './cf/0111_2k_cf/try_2k_0114full.json'
}
hparam_path = {
    'ike' : './hparams/IKE/',
    'rome': './hparams/ROME/',
    'memit': './hparams/MEMIT/',
}
model_alias = {
    'llama-2-7b-chat': ['llama-2-7b-chat', 'llama-7b-chat']
}
hparam_class= {
    'ike' :IKEHyperParams,
    'rome': ROMEHyperParams,
    'memit': MEMITHyperParams,
}
templates = [
    '''??? No, It can't be, This contradicts my understanding. It is pretty clear that the answer is to ' {prompt} ' is something else. Rethink and give the correct answer:''', # target_new
    '''? Really? But I am pretty sure that {prompt} {target_true}. Am I right? Yes! you are right. I apologize for any confusion. Reconsider that the answer is to 'question {prompt} ?' should be''' # prompt target_true
]

# 0111
# "induce": [], -- auto
# "api_rewrite_fillblank": [], -- done
# "wiki_pronoun_rewrite": [], -- done
# "wiki_pronoun_rewrite_irrelative": [], -- here with wiki
# "dialog_rewrite": [],  -- done
# "dialog_rewrite_insert": [],  -- done
# rewrite_keys = [
#     'api_rewrite_fillblank',
#     'wiki_pronoun_rewrite',
#     "wiki_pronoun_rewrite_irrelative",
#     'dialog_rewrite',
#     # 'wiki_anotherlong',
#     "dialog_rewrite_insert"
# ]

# 0114 
rewrite_keys = [
    'api_rewrite_fillblank',
    'wiki_rewrite_plain',
    'wiki_rewrite_pronoun',
    'wiki_rewrite_fillblank',
    'wiki_irre_rewrite_plain',
    'wiki_irre_rewrite_pronoun',
    'wiki_irre_rewrite_fillblank',
    'dialog_rewrite_plain',
    'dialog_rewrite_pronoun',
    'dialog_rewrite_fillblank',
    'dialog_rewrite_insert_plain',
    'dialog_rewrite_insert_pronoun',
    'dialog_rewrite_insert_fillblank',
]

# main
def add_profile(args):
    cache = jsonlines.open(args.save_metrics.split('.json')[0]+'_cache.jsonl', 'w')
    # sample profile
    with open(data_path[args.dataset], 'r') as f:
        data = json.load(f)
    # the_profile = profiling(args.profile_num, data)
    if args.profile_num:
        the_profile = data[:args.profile_num]
    else:
        the_profile = data

    hparam_ = hparam_path[args.edit] + args.base_model + '.yaml'
    hparams = hparam_class[args.edit].from_hparams(hparam_)
    if args.device != None:
        hparams.device = args.device
    
    device_name = 'cuda:' + str(hparams.device)
    if args.do_eval:
        orig_model_dir = hparams.model_name
        orig_model = LlamaForCausalLM.from_pretrained(orig_model_dir, cache_dir=args.cache_dir).to(device_name)
        tokenizer = LlamaTokenizer.from_pretrained(orig_model_dir, cache_dir=args.cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side='left'
    nvmetric_save = []
    editor = BaseEditor.from_hparams(hparams)
    for ip, prof in enumerate(tqdm(the_profile)):
        print('*'*20, ip, '*'*20)
        # if 'api_rewrite' in prof.keys() and len(prof['api_rewrite']) == 0:
        #     continue
        prompt, ground_truth, target_new, subject, paraphrase_prompt, generation_prompt = [], [], [], [], [], []
        prompt=[prof['requested_rewrite']['prompt'].format(prof['requested_rewrite']['subject'])]
        ground_truth=[prof['requested_rewrite']['target_true']['str']]
        target_new=[prof['requested_rewrite']['target_new']['str']]
        subject=[prof['requested_rewrite']['subject']]
        paraphrase_prompt=prof['paraphrase_prompts']
        generation_prompt=prof['generation_prompts']
        print('prompt', prompt)
        print('ground_truth', ground_truth)
        print('target_new', target_new)
        print('subject', subject)
        # no batch for now
        # compare two models
        # if ip > 0:
        #     params1 = list(editor.model.parameters())
        #     params2 = list(orig_model.parameters())
        #     for p1, p2 in zip(params1, params2):
        #         if torch.allclose(p1.data, p2.data):
        #             print("Parameters are equal.")
        #         else:
        #             print("Parameters are not equal.")
        edited_model = None
        gc.collect()
        torch.cuda.empty_cache()
        if ip > 0: # model changed
            editor.model = copy.deepcopy(orig_model)
            # editor.model = LlamaForCausalLM.from_pretrained(hparams.model_name, cache_dir=args.cache_dir).to(device_name)
        # params1 = list(editor.model.parameters())
        # params2 = list(orig_model.parameters())
        # for p1, p2 in zip(params1, params2):
        #     if torch.allclose(p1.data, p2.data):
        #         print("Parameters are equal.")
        #     else:
        #         print("Parameters are not equal.")
        metrics, edited_model, _  = editor.edit(
            prompts = prompt,
            ground_truth = ground_truth,
            target_new = target_new,
            subject = subject,
            keep_original_weight=False
        )
        print('metrics: ', metrics)
        # print('edited_model: ', type(edited_model))
        # need nvmetric
        if args.save:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            save_path = os.path.join(args.save_path, args.base_model)

            if os.path.exists(save_path) and bool(os.listdir(save_path)):
                os._exit(0)
            edited_model.save_pretrained(save_path)
            # tokenizer.save_pretrained(save_path)
        # if args.save_metrics:
        #     with open(args.save_metrics, 'w') as f:
        #         json.dump(nvmetric_save, f, indent=2)
        # evaluate the editing edited_model & model
        if args.do_eval:
            # prompt, rephrase
            prompts = prompt + paraphrase_prompt 
            batch = tokenizer(prompts, return_tensors='pt', padding=True)
            post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to(device_name),
                attention_mask=batch['attention_mask'].to(device_name),
                # max_length=60,
                max_new_tokens=30
            )
            # model = LlamaForCausalLM.from_pretrained(orig_model_dir, cache_dir='/data/maxb/mememe/EasyEdit/hugging_cache').to(device_name)
            # pre_edit_outputs = orig_model.generate(
            #     input_ids=batch['input_ids'].to(device_name),
            #     attention_mask=batch['attention_mask'].to(device_name),
            #     max_length=60,
            #     max_new_tokens=30
            # )
            # print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
            print('Post-Edit Outputs: ', [tokenizer.decode(x, skip_special_tokens = True) for x in post_edit_outputs.detach().cpu().numpy().tolist()])
            # post 
            generations = [tokenizer.decode(x, skip_special_tokens = True) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
            # generations = [ g[len("".join(icl_egs[:-1])):] for g in generations ]
            generations_ = [ generations[i].replace(prompts[i], '') for i in range(len(generations))]
            # score
            post_edit = {
                'short_generations': generations_,
                'generations': generations,
                'prompts': prompts,
                'score': {}
            }
            promptacc = 0
            rephacc = 0
            if target_new[0].lower() in generations[0].lower():
                promptacc += 1
            for ge in generations[1:]:
                if target_new[0].lower() in ge.lower():
                    rephacc += 1
            post_edit['score']['plain'] = promptacc
            post_edit['score']['rephrase'] =  rephacc/len(generations[1:])
            
            # induce
            induce_prompts = []
            for i in range(len(prompts)):
                # for j in range(len(templates)):
                induce_prompts.append(prompt[0] + generations_[i].split('\n')[0] +templates[0].replace('{target_new}', target_new[0]).replace('{prompt}', prompt[0]))
                induce_prompts.append(prompt[0] + generations_[i].split('\n')[0] +templates[1].replace('{prompt}', prompt[0]).replace('{target_true}', ground_truth[0]))
            batch = tokenizer(induce_prompts, return_tensors='pt', padding=True)
            induce_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to(device_name),
                attention_mask=batch['attention_mask'].to(device_name),
                # max_length=60,
                max_new_tokens=30
            )
            generations = [tokenizer.decode(x, skip_special_tokens = True) for x in induce_edit_outputs.detach().cpu().numpy().tolist()]
            # generations = [ g[len("".join(icl_egs[:-1])):] for g in generations ]
            generations_ = [ generations[i].replace(induce_prompts[i], '') for i in range(len(generations))]
            induceacc = 0
            for ge in generations_:
                if target_new[0].lower() in ge.lower():
                    induceacc += 1
            post_edit['score']['induce'] = induceacc/len(generations_)
            post_edit['induce'] = {
                'short_generations': generations_,
                'generations': generations,
                'prompts': induce_prompts,
            }
            
            # all the rewritten prompts
            if not args.dataset.startswith('counterfact'):
                for ri in rewrite_keys:
                    # rewrited = prof['api_rewrite']
                    # rewrited = prof['wiki_rewrite']
                    rewrited = prof[ri]
                    if type(rewrited) == str:
                        rewrited = [rewrited]
                    # rewrited = prof['wiki_another_rewrite']
                    if len(rewrited) == 0 or (rewrited[0]=='' and len(rewrited) == 1):
                        # rewrited = prompt
                        post_edit['score'][ri] = -1
                        post_edit[ri] = {
                            'short_generations': '',
                            'generations': '',
                            'prompts': '',
                        }
                        continue
                    batch = tokenizer(rewrited, return_tensors='pt', padding=True)
                    induce_edit_outputs = edited_model.generate(
                        input_ids=batch['input_ids'].to(device_name),
                        attention_mask=batch['attention_mask'].to(device_name),
                        # max_length=60,
                        max_new_tokens=30
                    )
                    generations = [tokenizer.decode(x, skip_special_tokens = True) for x in induce_edit_outputs.detach().cpu().numpy().tolist()]
                    # generations = [ g[len("".join(icl_egs[:-1])):] for g in generations ]
                    generations_ = [ generations[i][len(rewrited[i]):] for i in range(len(generations))]
                    rewacc = 0
                    for ge in generations_:
                        if target_new[0].lower() in ge.lower():
                            rewacc += 1
                    post_edit['score'][ri] = rewacc/len(generations_)
                    post_edit[ri] = {
                        'short_generations': generations_,
                        'generations': generations,
                        'prompts': rewrited,
                    }
                print(post_edit['score'])    
            # print(post_edit)
            post_edit['metrics'] = metrics
            nvmetric_save.append(post_edit)
            cache.write(post_edit)
    nvmetric_save_  = {
        'full_score': {},
    }
    for i in nvmetric_save[0]['score'].keys():
        the_scores = [mi['score'][i] for mi in nvmetric_save if mi['score'][i]!=-1]
        invalid = sum([1 for mi in nvmetric_save if mi['score'][i]==-1])
        nvmetric_save_['full_score'][i] = ("{:.4f}".format(sum(the_scores) / len(the_scores)), invalid)
    nvmetric_save_['samples'] = nvmetric_save
    if args.save_metrics:
        with open(args.save_metrics, 'w') as f:
            json.dump(nvmetric_save_, f, indent=2)
# entry 
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default='llama-2-7b-chat', type=str, help='')
parser.add_argument('--dataset', default='100cfapi70', type=str, help='')
parser.add_argument('--profile_num', default=None, type=int, help='')
parser.add_argument('--edit', default='rome', type=str, help='ike, rome, memit, all')
parser.add_argument('--device', default=None, type=int, help='')
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--paraphrase', action='store_true')
parser.add_argument('--neighborhood', action='store_true')
parser.add_argument('--attribute', action='store_true')
parser.add_argument('--generation', action='store_true')
parser.add_argument('--save_path', default='profile_save_edited', type=str, help='')
parser.add_argument('--save', action='store_true')
parser.add_argument('--save_metrics', default=None, type=str, help='')
parser.add_argument('--cache_dir', default=None, type=str, help='')
args = parser.parse_args()
add_profile(args)

#

# python baseline_loced_15.py --profile_num 2000 --edit memit --device 1 --dataset 2kfull --do_eval --save_metrics ./result_try/0111/100_memit.json
# python baseline_loced_15.py --profile_num 100 --edit rome --device 6 --dataset 2kfull --do_eval --save_metrics ./result_try/0111/100_rome.json
# python baseline_loced_15.py --profile_num 2000 --edit rome --device 0 --dataset 2kfull --do_eval --save_metrics ./result_try/0111/2k_rome.json

# python baseline_loced_0114.py --profile_num 2000 --edit memit --device 1 --dataset 2kfull0114 --do_eval --save_metrics ./result_try/0111/2kfull0114_memit.json
# python baseline_loced_0114.py --profile_num 2000 --edit rome --device 0 --dataset 2kfull0114 --do_eval --save_metrics ./result_try/0111/2kfull0114_rome.json

# python baseline_loced_0114.py --profile_num 1000 --base_model llama-2-13b-chat --edit memit --device 0 --dataset 2kfull0114 --do_eval --save_metrics ./result_try/0111/13b_2kfull0114_memit_con.json --start 748
# python baseline_loced_0114.py --profile_num 1000 --base_model llama-2-13b-chat --edit rome --device 1 --dataset 2kfull0114 --do_eval --save_metrics ./result_try/0111/13b_2kfull0114_rome.json