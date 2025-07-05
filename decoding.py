import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

class Decoding:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:

            # stop_word_ids = self.tokenizer.encode('\n' + stop_word)[2:] for Qwen 
            stop_word_ids = self.tokenizer.encode('\n' +stop_word)[3:]
            
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):

                
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens


            if mode == 'baseline':
                
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, do_sample=True, return_dict_in_generate=True,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
                
            elif mode == 'END':

                assert candidate_premature_layers is not None, "higher layers must be specified"

                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, 
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs)
                premature_layer_dist = outputs.premature_layer_dist
            sequences, scores = outputs.sequences, outputs.scores


            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh


    def extract_entropy(self,dict_outputs,indices,seq,candidate_premature_layers):
    
        dict_outputs = torch.stack([dict_outputs[i][0, :, :] for i in candidate_premature_layers],dim=0)

        if indices != None:
            dict_outputs = dict_outputs[:,:,indices]
        else:
            dict_outputs = dict_outputs[:,:,:]  
    
        distribution = []     
        distribution_tensor = torch.stack([dict_outputs[layer,seq,:] for layer in range(dict_outputs.shape[0])], dim=0)
        distribution_tensor = distribution_tensor.permute(1, 0) # shape: (selected_voc, num_layers) 
        probs = torch.softmax(distribution_tensor,dim=1)
        entropy = torch.distributions.Categorical(probs=probs, validate_args=False).entropy()
        return entropy
    

    def lm_score(self, input_text1, input_text2, en_alpha, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            
            if mode == 'baseline':
                
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)
                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            
            elif mode == 'END':

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                logits = outputs.logits[0]
                log_probs = torch.log_softmax(logits, dim=-1)

                premature_layer_dist[-1] = 0

                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]              
                final_logits = final_logits.log_softmax(dim=-1)
                diff_logits = final_logits
                
                for i in range(final_logits.shape[0]):

                    if relative_top > 0.0:
                        relative_top_mask = self.get_relative_top_filter(final_logits[i], relative_top)

                        false_indices = torch.nonzero(relative_top_mask == False,as_tuple=False).squeeze()
                        if false_indices.dim() == 0:
                            false_indices = false_indices.unsqueeze(0)

                        logits_values = diff_logits[i][false_indices].view(-1).cpu().numpy()
                        indices_values = false_indices.view(-1).cpu().numpy()                      
                        entropy = self.extract_entropy(dict_outputs,false_indices,i,candidate_premature_layers)   
                        diff_logits[i][false_indices] = diff_logits[i][false_indices]+ en_alpha*(-entropy)

                        if logits_values.shape != entropy.shape:
                            exit('entropy shape does not match!')
                        diff_logits[i] = diff_logits[i].softmax(dim=-1)
                        
                    else:
                        
                        logits_values = diff_logits[i].view(-1).cpu().numpy()
                        entropy = self.extract_entropy(dict_outputs,None,i,candidate_premature_layers)                           
                        diff_logits[i] = diff_logits[i]+ en_alpha*(-entropy)
                        
                        if logits_values.shape != entropy.shape:
                            
                            exit('entropy shape does not match!')

                        diff_logits[i] = diff_logits[i].log_softmax(dim=-1)


                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()



        return log_probs