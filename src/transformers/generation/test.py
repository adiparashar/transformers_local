# from .utils  import GenerationMixin
import os
from pathlib import Path
import random
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

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
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it").to('cuda')
    torch.device = 'cuda'
    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    data = random.sample(load_hf_data_set('validation','wmt14','de-en')['translation'],100)
    default_fwd_instruction = "Translate the following German sentence to English:"
    default_fwd_input_prefix = "German: "
    default_fwd_target_prefix = "English: "
    prompt_arr = [default_fwd_instruction,default_fwd_input_prefix]
    output_dict = {}
    for idx, d in enumerate(tqdm(data[:5], desc="Predicting")):
        prompt_arr.append(d['de'])
        prompt_arr.append(default_fwd_target_prefix)
        input_prompt = ('  ').join(prompt_arr)
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        breakpoint()
        outputs_arith = model.generate(
            input_ids = input_ids,
            # logits_processor=logits_processor,
            num_return_sequences = 5,
            do_sample = True,
            # stopping_criteria=stopping_criteria,
            max_new_tokens = 100,
            num_beams = 1,
            use_arithmetic = True
            )
        outputs_sample = model.generate(
            input_ids = input_ids,
            # logits_processor=logits_processor,
            num_return_sequences = 5,
            do_sample = True,
            # stopping_criteria=stopping_criteria,
            num_beams = 1,
            max_new_tokens = 100,
            use_arithmetic = False
            )
        output_dict[idx] = {}
        output_dict[idx]['arithmetic'] = tokenizer.batch_decode(outputs_arith, skip_special_tokens=True).split('English: ')[-1]
        output_dict[idx]['sampling'] = tokenizer.batch_decode(outputs_sample, skip_special_tokens=True).split('English: ')[-1]
        breakpoint()
    # input_prompt = "Today is a beautiful day, and"
    # input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
    # # breakpoint()
    # # instantiate logits processors
    # # logits_processor = LogitsProcessorList(
    # #     [
    # #         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    # #     ]
    # # )
    # # instantiate logits processors
    # # logits_warper = LogitsProcessorList(
    # #     [
    # #         TopKLogitsWarper(50),
    # #         TemperatureLogitsWarper(0.7),
    # #     ]
    # # )

    # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
    # # breakpoint()
    # torch.manual_seed(0)
    # # outputs = model.arithmetic_sample(
    # #     input_ids = input_ids,
    # #     logits_processor=logits_processor,
    # #     expand_size = 5,
    # #     # logits_warper=logits_warper,
    # #     stopping_criteria=stopping_criteria,
    # # )
    # # arithmetic sampling with num_return_sequences>1
    # outputs_arith = model.generate(
    #     input_ids = input_ids,
    #     logits_processor=logits_processor,
    #     num_return_sequences = 5,
    #     do_sample = True,
    #     # stopping_criteria=stopping_criteria,
    #     num_beams = 1,
    #     use_arithmetic = True
    # )
    # outputs_sample = model.generate(
    #     input_ids = input_ids,
    #     logits_processor=logits_processor,
    #     num_return_sequences = 5,
    #     do_sample = True,
    #     # stopping_criteria=stopping_criteria,
    #     num_beams = 1,
    #     use_arithmetic = False
    # )

    # print(f"Arithmetic sampling: {tokenizer.batch_decode(outputs_arith, skip_special_tokens=True)}")
    # print(f"Normals sampling: {tokenizer.batch_decode(outputs_sample, skip_special_tokens=True)}")
if __name__ == "__main__":
    test()
