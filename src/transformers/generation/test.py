# from .utils  import GenerationMixin
import json
import os
from pathlib import Path
import random
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList
from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
from collections import Counter

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import sys
import datasets
from src.transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from src.transformers.models.auto.tokenization_auto import AutoTokenizer
def load_hf_data_set(split,dataset_name, dataset_subname):
        # breakpoint()
        data = {}
        # datasets.config.DOWNLOADED_DATASETS_PATH = Path('/work/pi_dhruvesh_pate_umass_edu/aparashar_umass_edu/datasets')
        data[split] = datasets.load_dataset(dataset_name,dataset_subname, split="validation" )
        return data[split]
def test():
    print("this is is a test")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to('cuda')
    torch.device = 'cuda'
    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    data = random.sample(load_hf_data_set('validation','wmt14','de-en')['translation'],200)
    default_fwd_instruction = "Translate the following German sentence to an English sentence."
    default_fwd_input_prefix = "German sentence: "
    default_fwd_target_prefix = "English sentence: "
    prompt_arr = [default_fwd_instruction,default_fwd_input_prefix]
    output_dict = {}
    for idx, d in enumerate(tqdm(data, desc="Predicting")):
        prompt_arr.append(d['de'])
        prompt_arr.append(default_fwd_target_prefix)
        input_prompt = ('  ').join(prompt_arr)
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        # breakpoint()
        outputs_arith = model.generate(
            input_ids = input_ids,
            # logits_processor=logits_processor,
            num_return_sequences = 5,
            do_sample = True,
            # stopping_criteria=stopping_criteria,
            max_new_tokens = 100,
            temperature = 0.5,
            num_beams = 1,
            use_arithmetic = True
            )
        outputs_sample = model.generate(
            input_ids = input_ids,
            # logits_processor=logits_processor,
            num_return_sequences = 5,
            do_sample = True,
            temperature = 0.5,
            # stopping_criteria=stopping_criteria,
            num_beams = 1,
            max_new_tokens = 100,
            use_arithmetic = False
            )
        output_dict[idx] = {}
        output_dict[idx]['gt'] = d['en']
        output_dict[idx]['arithmetic'] = [i.split('English sentence: ')[-1].strip('\n') for i in tokenizer.batch_decode(outputs_arith, skip_special_tokens=True)]
        output_dict[idx]['sampling'] = [i.split('English sentence: ')[-1].strip('\n') for i in tokenizer.batch_decode(outputs_sample, skip_special_tokens=True)]
        output_dict[idx]['bleu_score_arith'], output_dict[idx]['n_gram_div_arith'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['arithmetic'])
        output_dict[idx]['bleu_score_sample'], output_dict[idx]['n_gram_div_sample'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['sampling'])
        
        # breakpoint()
    with open('flan_t5_wmt14_de-en_5__temp_0.5_output.json','a+') as f:
        json.dump(output_dict,f)
def calculate_bleu_and_ngram_diversity(reference, translations):
    bleu_score = corpus_bleu([reference]*len(translations), translations)

    n_values = [1, 2, 3, 4]
    total_unique_ngrams = 0
    ngram_diversity_score = 0
    for n in n_values:
        unique_ngrams = set()
        total_ngram_count = 0

        for translation in translations:
            # Compute n-grams
            # try:
            translation_ngrams = list(ngrams(translation.split(), n))
            # Count unique n-grams
            # breakpoint()

            total_ngram_count += len(list(translation_ngrams))
            unique_ngrams.update(translation_ngrams)
                # Count total n-grams
                
                # breakpoint()
            # except:
                # breakpoint()
                # continue
        # Update total counts
        total_unique_ngrams = len(unique_ngrams)
        ngram_diversity_score += total_unique_ngrams / total_ngram_count
        # breakpoint()
    return bleu_score, ngram_diversity_score

if __name__ == "__main__":
    test()
