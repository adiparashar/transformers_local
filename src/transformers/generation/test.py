# from .utils  import GenerationMixin
import os

from transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import sys

from src.transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
# from transformers_local.src.transformers.generation.logits_process import LogitsProcessorList, MinLengthLogitsProcessor

# from transformers.utils.dummy_pt_objects import AutoModelForCausalLM

# print the system path
print(sys.path)
from src.transformers.models.auto.tokenization_auto import AutoTokenizer
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     LogitsProcessorList,
#     MinLengthLogitsProcessor,
#     TopKLogitsWarper,
#     TemperatureLogitsWarper,
#     StoppingCriteriaList,
#     MaxLengthCriteria,
# )
def test():
    print("this is is a test")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").to('cuda')
    torch.device = 'cuda'
    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    input_prompt = "Today is a beautiful day, and"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
    # breakpoint()
    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ]
    )
    # instantiate logits processors
    # logits_warper = LogitsProcessorList(
    #     [
    #         TopKLogitsWarper(50),
    #         TemperatureLogitsWarper(0.7),
    #     ]
    # )

    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
    # breakpoint()
    torch.manual_seed(0)
    # outputs = model.arithmetic_sample(
    #     input_ids = input_ids,
    #     logits_processor=logits_processor,
    #     expand_size = 5,
    #     # logits_warper=logits_warper,
    #     stopping_criteria=stopping_criteria,
    # )
    # arithmetic sampling with num_return_sequences>1
    # outputs_arith = model.generate(
    #     input_ids = input_ids,
    #     logits_processor=logits_processor,
    #     num_return_sequences = 5,
    #     do_sample = True,
    #     # stopping_criteria=stopping_criteria,
    #     num_beams = 1,
    #     use_arithmetic = True
    # )
    outputs_sample = model.generate(
        input_ids = input_ids,
        logits_processor=logits_processor,
        num_return_sequences = 5,
        do_sample = True,
        # stopping_criteria=stopping_criteria,
        num_beams = 1,
        use_arithmetic = False
    )

    # print(f"Arithmetic sampling: {tokenizer.batch_decode(outputs_arith, skip_special_tokens=True)}")
    print(f"Normal sampling: {tokenizer.batch_decode(outputs_sample, skip_special_tokens=True)}")
if __name__ == "__main__":
    test()
