import os
from pathlib import Path
import random
from argparse import ArgumentParser

from utils import load_sentences


DS_ROOT = Path("data/") 
DATASET = os.environ.get('DATASET', 'enwik8_clean')

def parse_args():
    parser = ArgumentParser("Subsample dataset")
    parser.add_argument('--frac', type=float, default=0.1, help="Subsample relative size")
    parser.add_argument('--source-dataset',
                      type=Path,
                      default=DS_ROOT/DATASET,
                      help="Source dataset path")
    parser.add_argument('out_dataset',
                      type=Path,
                      help="Dest dataset path")
    args = parser.parse_args()
    return args
    

def main(args):
    sentences = load_sentences(args.source_dataset)
    sentences = random.sample(sentences, round(len(sentences) * args.frac))
    with open(args.out_dataset, 'w+') as fp:
        fp.write('.\n'.join(sentences))
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
