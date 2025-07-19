import torch
from transformers import set_seed
from lm_eval import tasks, utils
from evaluate import evaluate
from ppl_eval.ppl_eval import ppl_metric
from utils.customized_llama import LlamaForCausalLM, AutoTokenizer
from utils.utils import get_model
from box import Box


def main(args):
    model_name = args.base_model.split('/')[-1]
    args.log_file = open('{}_log.txt'.format(model_name), mode='a')
    print('load model...')
    lm = get_model(args)
    lm.model = lm.model.cpu()
    torch.cuda.empty_cache()
    del lm.model

    path = args.save_dir + '/{}/SR:{}_{}'.format(model_name, args.sparsity, model_name)

    model = LlamaForCausalLM.from_pretrained(path, device_map='auto', torch_dtype=torch.bfloat16)
    model.eval()
    lm.model = model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    args.model = lm
    print('start evaluate...')
    metric = ppl_metric(model, tokenizer, ['wikitext2'], 128, 2)
    mean, table = evaluate(lm, tokenizer, args)
    args.log_file.write('SR: {}, PPL: {}, Mean: {} \n {}\n'.format(args.sparsity, metric['wikitext2'], mean, table))
    args.log_file.flush()


if __name__ == "__main__":
    args = Box({
        "base_model": '7b',
        "sparsity": 0.0,
        "save_dir": "",
        "seed": 0,
        # eval args
        "tasks": None,
        "provide_description": False,
        "num_fewshot": 0,
        "batch_size": "32",
        "max_batch_size": None,
        "device": None,
        "output_path": None,
        "limit": None,
        "data_sampling": None,
        "no_cache": False,
        "model_args": '',
        "decontamination_ngrams_path": None,
        "description_dict_path": None,
        "check_integrity": False,
        "write_out": False,
        "output_base_path": None
    })
    print(args)
    set_seed(args.seed)
    model_name = 'Llama-{}'.format(args.base_model)
    args.model_args = "pretrained=meta-llama/{}".format(args.base_model)
    main(args) 