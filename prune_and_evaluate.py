import torch
import argparse
from transformers import set_seed
from tqdm import tqdm
import os
import copy
import pickle
from utils.utils import get_bookcorpus, get_alpaca, get_config, get_model
from utils.llama_utils import fast_OND, pruning
from utils.customized_llama import LlamaForCausalLM, AutoTokenizer
from ppl_eval.ppl_eval import ppl_metric
from lm_eval import tasks, utils
from evaluate import evaluate


def olica_pruning(model, dataloader, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    dtype = next(iter(model.parameters())).dtype
    model.eval()
    print("preparing...")
    layer_inputs = torch.zeros((len(dataloader), args.seqlen, model.config.hidden_size), dtype=dtype, device='cpu')
    inputs_len = []
    for i, data in tqdm(enumerate(dataloader), desc='Prepare input data.'):
        inp = data[0]
        inputs_len.append(args.seqlen)
        try:
            layer_inputs[i] = model.model.embed_tokens(inp)
        except ValueError:
            pass

    attention_mask = torch.ones((1, args.seqlen), dtype=torch.bool, device=layer_inputs.device)
    attention_mask_importance = model.model._prepare_decoder_attention_mask(attention_mask, (1, args.seqlen), layer_inputs, 0)
    layer_inputs_importance = layer_inputs
    position_ids_importance = torch.arange(args.seqlen, dtype=torch.long, device='cpu').unsqueeze(0)

    layer_inputs = copy.deepcopy(layer_inputs_importance)
    attention_mask = copy.deepcopy(attention_mask_importance)
    position_ids = copy.deepcopy(position_ids_importance)

    config = model.config
    model = model.cpu()
    torch.cuda.empty_cache()

    print("pruning...")
    init_inputs = (layer_inputs_importance, attention_mask_importance, position_ids_importance)
    mlp_r_list = fast_OND(model, init_inputs, dtype, config, args)
    mlp_index = mlp_r_list[:-1].argsort()[:args.mlp_num]

    init_inputs = (layer_inputs, attention_mask, position_ids)
    sparsity_qk, sparsity_vp, sparsity_mlp, sparsity, model = pruning(model, init_inputs, config, mlp_index, args)
    model.config.use_cache = use_cache
    return sparsity_qk, sparsity_vp, sparsity_mlp, sparsity, model


def main(args):
    model_name = args.base_model.split('/')[-1]
    args.log_file = open(f'{model_name}_log.txt', mode='a')
    print('load model...')
    # Load base model
    lm = get_model(args)
    lm.model = lm.model.cpu()
    torch.cuda.empty_cache()
    model = lm.model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Prepare dataset for pruning
    data_path = f'./data/{model_name}_{args.datasets}_nsample:{args.num_samples}_seqlen{args.seqlen}.pkl'
    dataloader = []
    if not os.path.exists(data_path):
        if 'bookcorpus' in args.datasets:
            dataloader += get_bookcorpus(nsamples=args.num_samples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
        if 'alpaca' in args.datasets:
            dataloader += get_alpaca(nsamples=args.num_samples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
        with open(data_path, 'wb') as f:
            pickle.dump(dataloader, f)
    else:
        with open(data_path, 'rb') as f:
            dataloader = pickle.load(f)

    num_samples = len(dataloader)
    if args.num_samples != num_samples:
        args.num_samples = num_samples
        print(f'{args.num_samples} datasets are sampled, args.num_samples is set to {args.num_samples}!')

    print('start olica pruning...')
    if max(args.sparsity, args.mlp_num) > 0:
        tick = torch.cuda.Event(enable_timing=True)
        tock = torch.cuda.Event(enable_timing=True)
        tick.record()
        sparsity_qk, sparsity_vp, sparsity_mlp, sparsity, pruned_model = olica_pruning(model, dataloader, args)
        tock.record()
        torch.cuda.synchronize()
        total_time = tick.elapsed_time(tock) / 1000.0
        s = f'total_time:{total_time:.2f}, sparsity_qk:{sparsity_qk:.2f}, sparsity_vp:{sparsity_vp:.2f}, sparsity_mlp:{sparsity_mlp:.2f}, Total sparsity:{sparsity:.2f}'
        print(s)
    else:
        pruned_model = model

    # Build pruned model in memory
    config = get_config(pruned_model)
    pruned_model = LlamaForCausalLM(config)
    pruned_model.half()
    pruned_model.load_state_dict(model.state_dict(), strict=True)
    pruned_model.eval()

    # Evaluate pruned model
    print('start evaluate...')
    metric = ppl_metric(pruned_model, tokenizer, ['wikitext2'], 128, 2)
    lm.model = pruned_model
    args.model = lm
    mean, table = evaluate(lm, tokenizer, args)
    args.log_file.write(f'SR: {args.sparsity}, PPL: {metric["wikitext2"]}, Mean: {mean} \n {table}\n')
    args.log_file.flush()
    print('Evaluation complete. Results logged.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default='7b', help="model to load")
    parser.add_argument("--datasets", type=str, default='bookcorpus+alpaca', help="Where to extract calibration data from.")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--seqlen", type=int, default=128, help="Sequence length for the calibration data.")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity ratio")
    parser.add_argument("--cache_dev", type=str, default="cuda", help="Defaults to `cuda`. When the GPU memory is insufficient, you can set `cache_dev` to `cpu`, but the trade-off is slower pruning speed.")
    parser.add_argument("--save_dir", type=str, default="", help="Path to saved model.")
    parser.add_argument("--percdamp", type=float, default=0.5, help="Percent of the average Hessian diagonal to use for dampening.")
    parser.add_argument("--ratio", type=float, default=0.03, help="Rank ratio for the SVD of linear calibration")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--mlp_num", type=int, default=0, help="Number of MLP layers to calibrate.")
    parser.add_argument("--k", type=float, default=3., help="ratio of QK vs VO.")
    # eval args
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=32)
    parser.add_argument("--max_batch_size", type=int, default=None, help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None, help="Limit the number of examples per task. If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--model_args", type=str, default='')
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    main(args) 