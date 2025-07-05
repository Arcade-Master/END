# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile


from decoding import Decoding


transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 7
COT_FLAG = True
# DEBUG = False
ANSWER_TRIGGER = "So the answer is"



def load_json(dataset_name,data_path, debug):

    input_file = data_path+dataset_name+".jsonl"
    # input_file = "./natural_questions.jsonl"
    print(input_file)
    with open(input_file, 'r') as f:
        data = f.read().strip()  
        if not data:
            raise ValueError("文件是空的")
        dataset = json.loads(data)

    list_data = list(dataset['question'])
    labels = list(dataset['answer'])

    if debug:
        list_data = list_data[0:20]
        labels = labels[0:20]

    return list_data, labels


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer



def build_prompt(question_text, prompt_style='zero_shot'):
    # this prompt is designed for trivia QA
    if prompt_style == 'zero_shot':
        question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt += f'Q:{question_text}\nA:'
    elif prompt_style == 'few_shot':
        # question_text_prompt = 'Answer the following question concisely.\n'
        question_text_prompt = f'Q: Who was President when the first Peanuts cartoon was published?\nA: Harry Truman\n\n'
        # question_text_prompt += f'Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nA: Sinclair Lewis\n\n'
        question_text_prompt += f'Q: Where in England was Dame Judi Dench born?\nA: York\n\n'
        question_text_prompt += f'Q: {question_text}\nA: '
    elif prompt_style == 'zero_shot_w_instru':
        raise NotImplementedError("zero_shot_w_instru Not implemented yet.")
    return question_text_prompt




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./")
    parser.add_argument("--output-path", type=str, default="./OGresult")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    # parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--relative_top", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dataset_name", type=str, choices=["triviaqa", "natural_questions", "hotpotqa"],
                        default="triviaqa")
    parser.add_argument("--prompt_style", type=str, choices=["zero_shot", "few_shot", "zero_shot_w_instru"],
                        default='few_shot')

    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # list_data_dict = load_csv(fp)
    list_data_dict, labels = load_json(args.dataset_name,args.data_path, args.debug)


    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = Decoding(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]

    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
    else:
        print(f"Entropy enhanced decoding with higher layers: {early_exit_layers[:-1]}")
        mode = "END"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    answers = []

    result_dict = {'qid_list': [], 'answers': {}, 'model_completion': {}, 'questions': {}}

    print("Begin inference...\n")
    # print("***Hyperparameters***:", args)
    print("\nSample prompt: \n", build_prompt(list_data_dict[0], args.prompt_style))
    print("*"*20)
    print("\n\n")


    for i, question in enumerate(tqdm(list_data_dict)):

        answer=labels[i]
        input_text=build_prompt(question, args.prompt_style)

        # input_text = build_prompt(sample)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers,en_alpha=args.alpha)
        
        model_completion = llm.generate(input_text, **generate_kwargs)

        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]

        if 'Q:' in model_completion:
            model_completion = model_completion.split('Q:')[0].strip()
        model_completion = model_completion.strip()


        print("-" * 20)
        print(f"Q{i}: {question}\nA: {answer}\nModel Response after processing: {model_completion}\n\n")

        # result_dict['model_completion'].append(model_completion)
        # result_dict['question'].append(sample)
        result_dict['qid_list'].append(i)
        result_dict['answers'][i] = answer
        result_dict['model_completion'][i] = model_completion
        result_dict['questions'][i] = question


        if args.debug:
            if i > 10:
                break

    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(result_dict, f)


    # evaluation
    if args.do_rating:
        from trivia_eval_util import evaluate_triviaqa
        from trivia_eval_util import evaluate_nq
        import json

        ground_truth = result_dict['answers']
        predicted_answers = result_dict['model_completion']
        qid_list = result_dict['qid_list']
        if args.dataset_name in ['triviaqa', 'hotpotqa']:
            eval_metrics = evaluate_triviaqa(ground_truth, predicted_answers, qid_list=qid_list, mute=False)
        elif args.dataset_name == 'natural_questions':
            eval_metrics = evaluate_nq(ground_truth, predicted_answers, qid_list=qid_list, mute=False)
        else:
            raise NotImplementedError(f"Dataset {args.dataset_name} not implemented yet.")

        # remove 'error_id' from eval_metrics
        if 'error_id' in eval_metrics:
            error_id_list = eval_metrics['error_id']
            del eval_metrics['error_id']
            eval_metrics['num_error'] = len(error_id_list)

            error_samples = {}
            for id in error_id_list:
                question = result_dict['questions'][id]
                answer = result_dict['answers'][id]['normalized_aliases'] if args.dataset_name == 'triviaqa' else \
                result_dict['answers'][id]
                prediction = result_dict['model_completion'][id]
                print(f"\n\nQ: {question}\nGT: {answer}\nA: {prediction}")
                error_sample = {'Q': question, 'model_prediction': prediction, 'A': answer, 'correct': 0}
                error_samples[id] = error_sample

            # record all the correct samples
            correct_samples = {}
            for id in qid_list:
                if id not in error_id_list:
                    question = result_dict['questions'][id]
                    answer = result_dict['answers'][id]['normalized_aliases'] if args.dataset_name == 'triviaqa' else \
                    result_dict['answers'][id]
                    prediction = result_dict['model_completion'][id]
                    # print(f"\n\nQ: {question}\nGT: {answer}\nA: {prediction}")
                    correct_sample = {'Q': question, 'model_prediction': prediction, 'A': answer, 'correct': 1}
                    correct_samples[id] = correct_sample

            final_samples = {'error_samples': error_samples, 'correct_samples': correct_samples}
            with open(output_file.replace('.json', '_results.json'), 'w') as f:
                json.dump(final_samples, f)

        exact_match_acc = eval_metrics['exact_match']
        f1 = eval_metrics['f1']
        print(f"acc:{exact_match_acc:.5f}\nf1:{f1:.5f}")
        print(args.output_path)

        # pdb.set_trace()
        eval_metrics['model_name'] = model_name
        eval_metrics['early_exit_layers'] = early_exit_layers
        eval_metrics['mode'] = mode
        # save all the paramters of args into eval_metrics
        eval_metrics['parameters'] = vars(args)
        eval_metrics['sample_prompt'] = build_prompt(list_data_dict[0], args.prompt_style)
        with open(output_file.replace('.json', '_rating.json'), 'w') as f:
            json.dump(eval_metrics, f)