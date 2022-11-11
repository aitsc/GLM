import sys, os
from tkinter import Y
sys.path.append(os.getcwd())

import data_utils
from tqdm import tqdm
from collections import Counter
from data_utils.tokenization import make_tokenizer
from tsc_draw import Draw
import json
from tasks.superglue.dataset import PROCESSORS
import math
from arguments import get_args
import numpy as np
from tasks.superglue.pvp import PVPS
import copy


def get_pt_token_num(path='../GLM/data/pretrain/bertbase/wikibook_glm.txt'):
    token_num_path = os.path.join(data_utils.lazy_loader.get_lazy_path(path), 'token_num.json')
    if os.path.exists(token_num_path):
        with open(token_num_path, 'r', encoding='utf8') as r:
            token_num = json.load(r)
    else:
        pre_tokenize = True  # 是否为token化之后的数据
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        text = data_utils.LazyLoader(
            path,
            data_type='text', 
            map_fn=map_fn, 
            mem_map=True,
            is_array=pre_tokenize,
            load_memory=True,  # 是否全部加载到内存
            half_load=False,  # 是否只加载一半
        )
        print('数据集token数量:', len(text.file))
        token_num = {str(k): int(v) for k, v in Counter(text.file).items()}
        with open(token_num_path, 'w', encoding='utf8') as w:
            json.dump(token_num, w)
    print('数据集token种数:', len(token_num))
    return token_num  # {'token':num,..}


def draw_token_num(token_num, split_pos, name, ylabel=None, cumsum=False):
    """绘制pdf图片

    Args:
        token_num (list): [('token',num),..]
        split_pos (list): [(start,end),..,[index,..],..]
        name (str): 文件名
        ylabel (list, optional): ['name',..]
        cumsum (bool, optional): 是否累计前面所有值
    """
    # 构建y值张量
    y_num_L = token_num
    if not isinstance(token_num[0][0], (tuple, list)):
        y_num_L = [token_num]
    x_token = [str(k) for k, v in y_num_L[0]]  # ['token_id',..]
    x_no_token = {v:k for k, v in enumerate(x_token)}  # {'token_id':序号,..}
    y_num_L = copy.deepcopy(y_num_L)
    for i in range(len(y_num_L)):
        y_num_L[i] = [int(v) for k, v in y_num_L[i]]
    y_num_L = np.array(y_num_L)  # [[num,..],..]
    if ylabel is None:
        ylabel = list(range(len(y_num_L)))
    # 初始化
    print('split_pos:', split_pos)
    sum_num = y_num_L.sum(-1)  # 每个数据集token数量
    draw = Draw(length=30, width=5 * len(split_pos), r=len(split_pos), c=1)
    for se in tqdm(split_pos, '绘图'):
        if len(se) == 2:
            s, e = se
            interval = math.ceil((e - s) / 200)
            xticks = ['' if i % interval else j for i, j in enumerate(x_token[s: e])]
            y = y_num_L[..., s: e]
        else:  # 按照id而不是范围取
            xticks = [str(i) for i in se]
            no_L = [x_no_token[str(i)] for i in se]
            y = y_num_L[..., no_L]
        # 统计
        current_sum_num = y.sum(-1)  # 当前每个数据集token数量
        max_y = y.max(-1)
        min_y = y.min(-1)
        max_divide_min = max_y / min_y
        max_divide_min[np.isinf(max_divide_min)] = 0
        proportion_of_total = current_sum_num / sum_num
        # 标题
        title=f'rank_range=[{s},{e}); max/min=max/min,proportion_of_total'
        if len(se) == 2 and s == 0 and e == len(y_num_L[0]):
            title += f',all_total'
        ylabel_ = []
        for i in zip(ylabel, max_y, min_y, max_divide_min, proportion_of_total, sum_num):
            ylabel_.append(str(i[0]) + ':%d/%d=%.1f,%.2e' % i[1:-1])
            if len(se) == 2 and s == 0 and e == len(y_num_L[0]):
                ylabel_[-1] += ',%d' % i[-1]
        # cumsum
        y_left = y[0:1]
        y_right = (y[1:] / current_sum_num[1:, None]) if len(y) > 1 else None
        if cumsum:
            y_left = np.cumsum(y_left, axis=-1)
            if y_right is not None:
                y_right = np.cumsum(y_right, axis=-1)
        draw.add_line(
            x=list(range(len(xticks))),
            xaxis='token',
            y_left=y_left.tolist(),
            yaxis_left='num',  # 第一个用值, x顺序是按第一个排序的
            ylabel_left=ylabel_[0:1],
            y_right=y_right.tolist() if len(y) > 1 else None,
            yaxis_right='percentage',  # 后面的在right轴上用当前的百分比
            ylabel_right=ylabel_[1:],
            x_rotation=90,
            xticks=xticks,
            markersize=1,
            lw=0.5,
            title=title,
            annotate=False,
        )
    draw.draw(f'tsc/ah_tokens_{name}.pdf')

# tokenizer
tokenizer = make_tokenizer('BertWordPieceTokenizer', None, 'tokenizer.model', 30522,
                            'bert-base-uncased', add_block_symbols=True, cache_dir=None,
                            add_sentinel_token=0, add_task_mask=False,
                            add_decoder_mask=False,
                            fix_command_token=False)
save_token_id = {i.Id for i in tokenizer._command_tokens}
pvp_token = set()
for pvp in PVPS.values():  # 获取 VERBALIZER
    for v in ['VERBALIZER', 'VERBALIZER_A', 'VERBALIZER_B']:
        if not hasattr(pvp, v):
            continue
        bar = getattr(pvp, v)
        if isinstance(getattr(pvp, v), dict):
            bar = bar.values()
        for tokens in bar:
            for token in tokens:
                pvp_token.add(token.strip())
                pvp_token.add(token.strip().lower())
print(f'pvp_token({len(pvp_token)}):', sorted(pvp_token))
pvp_token_no_id = set()  # 无法 tokenizer 的token词
for t in pvp_token:
    try:
        save_token_id.add(tokenizer.TokenToId(t))
    except:
        pvp_token_no_id.add(t)
print(f'pvp_token_no_id({len(pvp_token_no_id)}):', sorted(pvp_token_no_id))
print(f'这些token不能省略({len(save_token_id)}):', sorted(save_token_id))
unk_id = tokenizer.get_command('unk').Id
print('unk_id:', unk_id)

# 预训练语料
token_num = get_pt_token_num()  # {'token':num,..}
[token_num.setdefault(str(i), 0) for i in range(tokenizer.num_tokens)]  # 填充0次出现的token
token_num_sort = sorted(token_num.items(), key=lambda t:t[1], reverse=True)  # 排序
split_pos = [(20000, tokenizer.num_tokens)]  # 分段绘制
split_pos += [(0, tokenizer.num_tokens), [0, 0]]  # 全段 + 每100倍换一段
for i, (token, num) in enumerate(token_num_sort):
    if num and num * 100 < token_num_sort[split_pos[-1][0]][1]:
        split_pos[-1][1] = i
        split_pos.append([i, tokenizer.num_tokens])
split_pos.append([i[0] for i in token_num_sort if int(i[0]) in save_token_id])  # 不能省略的token

def get_ft_token_num(data_dir, processor, tokenizer, args, split='train', token_num_sort=None):
    # 获取微调语料的数据
    token_num_path = os.path.join(data_dir, f'{split}_token_num.json')
    if os.path.exists(token_num_path):
        with open(token_num_path, 'r', encoding='utf8') as r:
            return json.load(r)
    if split == 'train':
        examples = processor.get_train_examples(data_dir)
    elif split == 'dev':
        examples = processor.get_dev_examples(data_dir)
    sample = [processor.encode(i, tokenizer, args.seq_length, args) for i in examples]
    tokens = np.concatenate([i['text'].flatten() for i in sample], axis=0)
    c_id = {i.Id for i in tokenizer._command_tokens}  #  去除 command_tokens
    token_num_ = {str(k): 0 if int(k) in c_id else int(v) for k, v in Counter(tokens).items()}
    if token_num_sort:
        token_num_ = [(k, token_num_[k] if k in token_num_ else 0) for k, v in token_num_sort]
    with open(token_num_path, 'w', encoding='utf8') as w:
        json.dump(token_num_, w)
    return token_num_  # {'token':num,..} or [('token',num),..]

# 微调语料
task_path = '../GLM/data/english_data/superglue'
token_num_L = [token_num_sort]
ylabel = ['pretrain']
args = get_args()
args.seq_length = args.max_position_embeddings = 512
for task in tqdm(['WiC', 'RTE', 'CB', 'WSC', 'BoolQ', 'COPA', 'MultiRC', 'ReCoRD'], '微调数据'):
    processor = PROCESSORS[task.lower()](args)
    data_dir = os.path.join(task_path, task)
    token_num_train = get_ft_token_num(data_dir, processor, tokenizer, args, split='train', token_num_sort=token_num_sort)
    token_num_dev = get_ft_token_num(data_dir, processor, tokenizer, args, split='dev', token_num_sort=token_num_sort)
    token_num_L.append(token_num_train)
    token_num_L.append(token_num_dev)
    ylabel.append(task + '_train')
    ylabel.append(task + '_dev')

# draw_token_num(token_num_sort, split_pos, 'wikibook_glm')  # 预训练绘图
# draw_token_num(token_num_L, split_pos, 'all', ylabel)  # 微调绘图(左侧预训练数值,右侧微调百分比)
draw_token_num(token_num_L, split_pos, 'all_cumsum', ylabel, True)  # 微调绘图, 累计(左侧预训练数值,右侧微调百分比)
