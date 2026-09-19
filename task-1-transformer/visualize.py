"""从已有 checkpoint 生成注意力热图,无需重新训练。

用法:
    python visualize.py                      # 自动取最新的 ckpt/tsf_*.pt
    python visualize.py --ckpt ckpt/tsf_9.pt # 指定 checkpoint

复用 train.py 中的词表构建 / 样本挑选 / 绘图函数(train.py 的 main() 受
`if __name__=='__main__'` 保护,import 时不会触发训练)。
"""
import argparse
import glob
import os

import pandas as pd
import torch

from src.model import TransformerClassifier
import train


def latest_ckpt(pattern='ckpt/tsf_*.pt'):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'没有找到 checkpoint: {pattern}')

    def epoch_of(f):
        base = os.path.splitext(os.path.basename(f))[0]
        try:
            return int(base.split('_')[-1])
        except ValueError:
            return -1

    return max(files, key=epoch_of)


def main():
    ap = argparse.ArgumentParser(description='从 checkpoint 生成注意力热图')
    ap.add_argument('--ckpt', default=None, help='checkpoint 路径,默认取最新的 ckpt/tsf_*.pt')
    ap.add_argument('--train_dir', default='data/train.parquet', help='用于重建词表(需与训练一致)')
    ap.add_argument('--sample_dir', default='data/test.parquet', help='挑选可视化样本的数据集')
    ap.add_argument('--out', default='figures')
    ap.add_argument('-dmodel', type=int, default=128)
    ap.add_argument('-dff', type=int, default=512)
    ap.add_argument('-nh', type=int, default=4)
    ap.add_argument('-l', type=int, default=9)
    ap.add_argument('-v', type=int, default=2500)
    ap.add_argument('-max', type=int, default=512)
    args = ap.parse_args()

    ckpt = args.ckpt or latest_ckpt()

    # 词表必须用与训练相同的数据/参数重建,才能对齐 embedding 索引
    df = pd.read_parquet(args.train_dir)
    vocab, word2idx, idx2word = train.build_personal_vocab(df['text'].tolist(), vocab_len=args.v)

    model = TransformerClassifier(args.dmodel, args.nh, args.dff, word2idx,
                                  N=args.l, vocab_len=args.v, max_len=args.max)
    state = torch.load(ckpt, map_location=train.device)
    model.load_state_dict(state)
    model = model.to(train.device)
    model.eval()
    print(f'loaded {ckpt}')

    cjk = train.setup_cjk_font()
    if cjk is None:
        print('[warn] 未找到中文字体,坐标轴将用位置索引。可安装 fonts-noto-cjk 后重试。')
    else:
        print(f'using CJK font: {cjk}')

    os.makedirs(args.out, exist_ok=True)
    samples = train.pick_viz_samples(pd.read_parquet(args.sample_dir))
    if not samples:
        print('[warn] 未挑到合适样本,请检查数据集的 label/text 字段。')
    for tag, text in samples:
        saved = train.plot_attention_grid(model, text, tag, args.l, args.nh,
                                          out_dir=args.out, cjk_ok=(cjk is not None))
        print(f'saved {saved}  ({tag}: {text[:20]})')


if __name__ == '__main__':
    main()
