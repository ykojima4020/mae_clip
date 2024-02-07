import sys
sys.path.append('../')

from tqdm import tqdm

from transformers import DistilBertTokenizer

from mae_clip_training import get_args_parser
from misc.config import get_config

from data.dataloader_builder import GCC3MDataLoaderBuilder

args = get_args_parser()
args = args.parse_args()
cfg = get_config(args)

tokenizer = DistilBertTokenizer.from_pretrained(cfg.model.text.encoder.name) 
gcc3m_dataloader_builder = GCC3MDataLoaderBuilder(cfg.data, tokenizer, cfg.data.batch_size, cfg.data.num_workers)

train_loader = gcc3m_dataloader_builder('train', test=cfg.test)

n_data = 0
for batch in tqdm(train_loader):
    n_data += len(batch['image'])

print(n_data)


