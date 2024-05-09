# from .utils  import GenerationMixin
import json
import os
from pathlib import Path
import random
import time
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams
from collections import Counter
from accelerate import Accelerator
from accelerate.utils import gather_object

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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").cuda()
    accelerator = Accelerator()
    if False:
            # breakpoint()
            bnb_config= BitsAndBytesConfig(
            load_in_8bit=True,)
            self.model = cls_model.from_pretrained(model,
                                                torch_dtype=torch.bfloat16,
                                                device_map={"": accelerator.process_index},
                                                quantization_config=bnb_config,
                                                # low_cpu_mem_usage=low_cpu_mem_usage,
                                                cache_dir = '/work/pi_mccallum_umass_edu/aparashar_umass_edu/models/.cache',
                                                trust_remote_code=True,
                                                )
    model = model.bfloat16().cuda()
    # torch.device = 'cuda'
    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    data = random.sample(load_hf_data_set('validation','wmt14','de-en')['translation'],2)
    # default_fwd_instruction = "Translate the following German sentence to an English sentence."
    # default_fwd_input_prefix = "German sentence: "
    # default_fwd_target_prefix = "English sentence: "
    # prompt_arr = [default_fwd_instruction,default_fwd_input_prefix]
    default_fwd_input_prefix = """Like each of the previous examples, answer the question with either "Yes" or "No".
    Provide reasoning for your answer.
    End your reasoning with the sentence: "So the answer is <your_answer>". Replace "<your_answer>" with your final answer, which should be either "Yes" or "No"."""
    turn_prefix_model = '<start_of_turn>model'
    turn_prefix_human = '<start_of_turn>human'
    turn_suffix = '<end_of_turn>'
    default_fwd_demos_prefix = "Here are some examples of reasoning based question answers."
    default_fwd_reasoning_prefix = "Reasoning: "
    default_fwd_question_prefix = "Question: "
    default_fwd_target_prefix = "Answer: "
    qa_exemplars = [
            {
                "question": "Do hamsters provide food for any animals?",
                "reasoning": "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.",
                "answer" : "Yes"
            },
            {
                "question": "Could Brooke Shields succeed at University of Pennsylvania?",
                "reasoning": "Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.",
                "answer" : "Yes"
            },
            {
                "question": "Hydrogen’s atomic number squared exceeds number of Spice Girls?",
                "reasoning": "Hydroßen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is no.",
                "answer" : "No"
            },
            {
                "question": "Is it common to see frost during some college commencements?",
                "reasoning": "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.",
                "answer" : "Yes"
            },
            {
                "question": "Could a llama birth twice during War in Vietnam (1945-46)?",
                "reasoning": "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.",
                "answer" : "No"
            },
            {
                "question": "Would a pear sink in water?",
                "reasoning": "The density of a pear is about 0.6 g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.",
                "answer" : "No"
            }
        ]

    
    demos = random.sample(qa_exemplars,6)
    demos = [[default_fwd_question_prefix + d["question"]]+ [default_fwd_reasoning_prefix + d["reasoning"] + default_fwd_target_prefix + d["answer"]] for d in demos]# def test(prompt_arr, model, tokenizer, data,demos, default_fwd_target_prefix, N = 10, temp = 0.5):

    
    output_dict = {}
    data =  [
        {
      "input": "Gandalf hypothetically defeats Rincewind in a wizard battle?",
      "target_scores": {
        "Yes": 1,
        "No": 0
      },
      "target": "Yes. Gandalf is a 2000 plus year old wizard that has fought orcs and spirits in Middle Earth. Rincewind is the protagonist of the Discworld series. Rincewind is a failed student at the Unseen University for wizards in Ankh-Morpork. Rincewind is described by other wizards as the magical equivalent to the number zero.",
       },
        {
            "input": "Does ontology require a scalpel?",
      "target_scores": {
        "Yes": 0,
        "No": 1
      },
      "target": "No. A scalpel is used during surgery. Ontology is a philosophical domain, not a medical one.",
       },
       {"input": "Is it normal to find parsley in multiple sections of the grocery store?",
      "target_scores": {
        "Yes": 1,
        "No": 0
      },
      "target": "Yes. Parsley is available in both fresh and dry forms. Fresh parsley must be kept cool. Dry parsley is a shelf stable product.",
      },{
        "input": "Has Johns Hopkins University always treated subjects ethically?",
      "target_scores": {
        "Yes": 0,
        "No": 1
      },
      "target": "No. Henrietta Lacks' DNA was used and replicated by Johns Hopkins University without her family's knowledge or approval. Henrietta Lacks' family medical history was released by Johns Hopkins University without their knowledge.",
      
      }]
    for idx, d in enumerate(tqdm(data, desc="Predicting")):
        # prompt_arr.append(d['de'])
        # prompt_arr.append(default_fwd_target_prefix)
        # input_prompt = ('  ').join(prompt_arr)

        prompt_arr = [default_fwd_input_prefix] +[default_fwd_question_prefix + d['input']]
        prompt_arr = demos + [prompt_arr]
        prompt_arr = [element for sublist in prompt_arr for element in sublist]
        # print(prompt_arr)
        # input_prompt = turn_prefix_human + (' \n').join(prompt_arr) + turn_suffix + ' ' + turn_prefix_model
        input_prompt = (' \n').join(prompt_arr)
    
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        # breakpoint()
        num_return_sequences_per_gpu = 80 // accelerator.num_processes

# Divide the total number of sequences across GPUs
        with accelerator.split_between_processes(range(accelerator.num_processes)) as idx:
            # Store outputs of generations in a list
            outputs = []

            # Have each GPU generate sequences
            for i in idx:
                # Generate sequences on the current GPU
                output = model.generate(
                    input_ids=input_ids,
                    num_return_sequences=num_return_sequences_per_gpu,
                    do_sample=True,
                    num_beams=1,
                    max_new_tokens=100,
                    use_arithmetic = True
                )

                # Decode and store the generated sequences
                decoded_outputs = [tokenizer.decode(sequence) for sequence in output]
                with open('output.txt','a+') as f:
                    f.write("Decoded outputs")
                    f.write(str('\n'.join(decoded_outputs)))
                    f.write("\n")
                outputs.extend(decoded_outputs)
                with open('output.txt','a+') as f:
                    f.write(f"outputs{idx}")
                    f.write(str('\n'.join(outputs)))
                    f.write("\n")

                breakpoint()
        # Gather the generated sequences from all GPUs

        gathered_outputs = accelerator.gather(outputs)
        with open('output.txt','a+') as f:
                    f.write("gathered_outputs")
                    f.write(str('\n'.join(gathered_outputs)))
                    f.write("\n")
        arithttime =time.time()
        # outputs_arith = model.generate(
        #     input_ids = input_ids,
        #     # logits_processor=logits_processor,
        #     num_return_sequences = 10,
        #     do_sample = True,
        #     # stopping_criteria=stopping_criteria,
        #     max_new_tokens = 100,
        #     num_beams = 1,
        #     use_arithmetic = True
        #     )
        # arithttime2 =time.time()-arithttime
            
        # # outputs_arith = model.generate(
        # #     input_ids = input_ids,
        # #     # logits_processor=logits_processor,
        # #     num_return_sequences = 60,
        # #     do_sample = True,
        # #     # stopping_criteria=stopping_criteria,
        # #     max_new_tokens = 100,
        # #     num_beams = 1,
        # #     use_arithmetic = True
        # #     )
        # sample_time = time.time()
        # outputs_sample = model.generate(
        #     input_ids = input_ids,
        #     num_return_sequences = 10,
        #     do_sample = True,
        #     num_beams = 1,
        #     max_new_tokens = 100,
        #     use_arithmetic = False
        #     )
        # sample_time2 = time.time()-sample_time
        # breakpoint()
        # output_dict[idx] = {}
        # output_dict[idx]['gt'] = d['en']
        # output_dict[idx]['arithmetic'] = [i.split('English sentence: ')[-1].strip('\n') for i in tokenizer.batch_decode(outputs_arith, skip_special_tokens=True)]
        # output_dict[idx]['sampling'] = [i.split('English sentence: ')[-1].strip('\n') for i in tokenizer.batch_decode(outputs_sample, skip_special_tokens=True)]
        # output_dict[idx]['bleu_score_arith'], output_dict[idx]['n_gram_div_arith'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['arithmetic'])
        # output_dict[idx]['bleu_score_sample'], output_dict[idx]['n_gram_div_sample'] = calculate_bleu_and_ngram_diversity(output_dict[idx]['gt'], output_dict[idx]['sampling'])
        
        # breakpoint()
    # with open('flan_t5_wmt14_de-en_10__temp_1.0_output.json','a+') as f:
    #     json.dump(output_dict,f)
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
        try:
            total_unique_ngrams = len(unique_ngrams)
            ngram_diversity_score += total_unique_ngrams / total_ngram_count
        except:
            print(translation_ngrams)
            continue
        # breakpoint()
    return bleu_score, ngram_diversity_score

if __name__ == "__main__":
    test()
    
