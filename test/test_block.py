import sys, os
sys.path.append(os.getcwd())

import random
import numpy as np
from blocklm_utils import ConstructBlockStrategy
from argparse import Namespace


# rng = random.Random()
# span_lengths = [2, 3, 4, 2, 3, 4]
# length = 100
#
# counts = np.array([0] * length)
# for _ in range(10000):
#     rng.shuffle(span_lengths)
#     spans = ConstructBlockStrategy.sample_spans(span_lengths, length, rng)
#     for start, end in spans:
#         counts[start: end] += 1
# print(counts)
def main():
    args = Namespace()
    args.seq_length = 10
    args.eod_token = 0

    strategy = ConstructBlockStrategy(args, None, bert_ratio=0.4, max_seq_length=128)
    counts = np.array([0] * 10)
    for _ in range(10000):
        spans = strategy.sample_span_in_document(np.array([1, 2, 3, 0, 4, 5, 6, 7, 9, 0], dtype=np.long), [1, 1],
                                                 random.Random())
        for start, end in spans:
            counts[start: end] += 1

    print(counts)


if __name__ == '__main__':
    import mpu
    import torch
    # mpu 依赖初始化的 torch.distributed
    init_method = 'tcp://localhost:6002'
    torch.distributed.init_process_group(
        backend='nccl',  # 指定当前进程要使用的通信后端
        world_size=1,  # 该 job 中的总进程数。如果指定 store 参数，则需要指定该参数
        rank=0,  # 表示当前进程的编号，即优先级。如果指定 store 参数，则必须指定该参数. rank=0 的为主进程，即 master 节点
        init_method=init_method,  # 指定当前进程组初始化方式。如果未指定 init_method 及 store，则默认为 env://，表示使用读取环境变量的方式进行初始化。该参数与 store 互斥
    )
    # 初始化 mpu
    mpu.initialize_model_parallel(1)
    main()