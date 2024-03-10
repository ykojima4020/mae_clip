# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# %%
# %reset -f

# %%
import torch

import sys
sys.path.append('../')

from factory import RILSMAECLIPFactory, PretrainedOpenCLIPFactory
from evaluator.evaluator import ZeroShotImageNetEvaluator

# %%
from misc.config import load_config
from omegaconf import OmegaConf

config = '../config/ttt_mae.yaml'
cfg = load_config(config)
OmegaConf.set_struct(cfg, True)

model_path = '../output/20240304_decoder_layer8_finetune_datacomp_l_s1b-b8k/checkpoint.pth'

device = 'cuda'

# %%
factory = PretrainedOpenCLIPFactory(cfg.model)
model, tokenizer, transform = factory.create()
model = model.to(device)

status = torch.load(model_path, map_location="cuda")

model.load_state_dict(status['model'])
model.eval()

# %%
evaluator = ZeroShotImageNetEvaluator(tokenizer, device, transform('valid'))
eval_res = evaluator(model.clip, update=False)
print(eval_res)

# %%
eval_res = evaluator(model.clip, update=False)
print(eval_res)

# %%
