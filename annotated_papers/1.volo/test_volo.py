import os

import torch
from volo import volo_d1_224, volo_d2_224, volo_d3_224, volo_d4_224, volo_d5_224

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from tabulate import tabulate
from timm import create_model
from deepspeed.profiling.flops_profiler import get_model_profile


def compute_flops(model):
    return get_model_profile(
        model=model.cuda(),
        input_res=(1, 3, 224, 224),
        print_profile=False,
        detailed=False,
        warm_up=10,
        as_string=False,
        output_file=None,
        ignore_modules=None
    )[1] / (1024 * 1024 * 1024)


def compute_param(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad]) / (1024 * 1024)

if __name__ == '__main__':
    results = []

    for model_fn in [volo_d1_224, volo_d2_224, volo_d3_224, volo_d4_224, volo_d5_224]:
        # To match with Timm version, you should set `use_token_label=True`
        model = model_fn(use_token_label=True)
        x = torch.rand([2, 3, 224, 224])
        y = model(x)
        flops = compute_flops(model)
        params = compute_param(model)
        results.append([f'(ours){model_fn.__name__}', round(flops, 2), round(params, 2)])

    for model_name in ['volo_d1_224', 'volo_d2_224', 'volo_d3_224', 'volo_d4_224', 'volo_d5_224']:
        model = create_model(model_name)
        x = torch.rand([2, 3, 224, 224])
        y = model(x)
        flops = compute_flops(model)
        params = compute_param(model)
        results.append([f'(timm){model_name}', round(flops, 2), round(params, 2)])

    print(tabulate(results, headers=['model', 'flops(G)', 'param(M)']))
