import torch
import os
import re
from pprint import pprint
from collections import OrderedDict
import json


def getsize(path):
    return re.sub('([0-9]{1,4})', r'\1,', str(os.path.getsize(path))[::-1])[::-1][1:] + ' 字节'

def size_to_int(size):
    return int(size.replace(',', '').split(' ', 1)[0])


test_pt = 'tsc/test.pt'
# 递归拆解
def get_pt_info(pt_dict, save_low_byte=1e8):
    if type(pt_dict) in {list, tuple}:
        return [get_pt_info(v, save_low_byte) for v in pt_dict]
    if type(pt_dict) not in {dict, OrderedDict}:
        v = pt_dict
        with open(test_pt, 'wb') as w:
            torch.save({'': v}, w)
        if size_to_int(getsize(test_pt)) < save_low_byte:
            return None
        return {
            '*size': getsize(test_pt),
            '*type': re.search("(?<=').+(?=')", str(type(v))).group(),
            **({'*shape': list(v.shape), '*dtype': str(v.dtype)} if type(v) == torch.Tensor else {}),
        }
    pt_info = {}
    for k, v in pt_dict.items():  # k 参数名 v 对应参数值
        with open(test_pt, 'wb') as w:
            torch.save({k: v}, w)
        if size_to_int(getsize(test_pt)) < save_low_byte:
            continue
        pt_info[k] = {
            '*size': getsize(test_pt),
            '*type': re.search("(?<=').+(?=')", str(type(v))).group(),
            **({'*shape': list(v.shape), '*dtype': str(v.dtype)} if type(v) == torch.Tensor else {}),
            **({'*keys': get_pt_info(v, save_low_byte)} if type(v) in {dict, OrderedDict} else {}),
            **({'*list': get_pt_info(v, save_low_byte)} if type(v) in {list, tuple} else {}),
        }
    return pt_info


def get_pt_key(pt_dict):
    d = {}
    if type(pt_dict) not in {dict, OrderedDict}:
        return d
    for k, v in pt_dict.items():
        d[k] = get_pt_key(v)
    return d


path_info = OrderedDict()
save_low_byte = 1e8  # 小于这个字节的文件不再递归
# save_low_byte = 0
for path in [
    'data/checkpoints/pretrain/blocklm-base-blank/150000/mp_rank_00_model_states.pt',  # 218,710,529 字节
    'data/checkpoints/other/student-pre-tiny-64*100000/100000/mp_rank_00_model_states.pt',  # 935,392,964 字节
    'data/checkpoints/pretrain/block_base/blocklm-blank07-25-13-30/390000/mp_rank_00_model_states.pt',  # 1,312,226,109 字节
    'data/checkpoints/pretrain/block_base/blocklm-blank07-23-14-42/140000/mp_rank_00_model_states.pt',  # 1,530,788,676 字节
    'data/checkpoints/pretrain/block_base/test/torch.distributed-gpu2mp2/200/mp_rank_01_model_states.pt',
    'data/checkpoints/pretrain/block_base/test/deepspeed-gpu2mp2/300/mp_rank_01_model_states.pt',
]:
    print(getsize(path), ':', path)
    pt_dict = torch.load(path)
    pt_info = get_pt_info(pt_dict, save_low_byte=save_low_byte)
    path_info[path] = {
        '*keys': pt_info,
        '*size': getsize(path),
    }
    pprint(get_pt_key(pt_dict), width=150)
    print()

with open(test_pt + '.json', 'w', encoding='utf8') as w:
    json.dump(path_info, w, indent=2, ensure_ascii=False)
