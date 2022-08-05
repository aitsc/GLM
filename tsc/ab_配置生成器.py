from datetime import datetime
import re

class Models:
    @staticmethod
    def model_blocklm_base(env: dict, **kw):
        env['MODEL_TYPE'] = "blank-base"
        env['MODEL_PATH'] = "data/checkpoints/pretrain/blocklm-base-blank"  # 官方模型
        # env['MODEL_PATH'] = "data/checkpoints/pretrain/block_base/blocklm-blank07-23-14-42"  # fp16-books1-64*140000
        # env['MODEL_PATH'] = "data/checkpoints/pretrain/block_base/blocklm-blank07-24-09-23"  # fp32-books1-64*140000
        # env['MODEL_PATH'] = "data/checkpoints/pretrain/block_base/blocklm-blank07-25-13-30"  # fp32-wiki+books1-64*390000
        env['MODEL_ARGS'] = [
            '--block-lm', 
            '--num-layers 12', 
            '--hidden-size 768', 
            '--num-attention-heads 12', 
            '--max-position-embeddings 512', 
            '--tokenizer-model-type bert-base-uncased', 
            '--tokenizer-type BertWordPieceTokenizer', 
            '--load-pretrained ' + env['MODEL_PATH'],
            '--fp16',
            # '--fp32-allreduce',
        ]
        return env

    @staticmethod
    def block_tiny6(env: dict, **kw):
        env['MODEL_TYPE'] = "blank-tiny6"
        env['MODEL_PATH'] = "data/checkpoints/pretrain/block_tiny6/blocklm-blank07-31-07-36"  # tiny6(fp16)+wiki(15G) 128*285000
        env['MODEL_PATH'] = "data/checkpoints/other/student-em+pre6-64*100"  # distil6(fp16)+wiki(15G)（预训练蒸馏 em+pre）
        env['MODEL_PATH'] = "data/checkpoints/other/student-em+pre6-64*100000"  # distil6(fp16)+wiki(15G)（预训练蒸馏 em+pre）
        env['MODEL_PATH'] = "data/checkpoints/other/tiny6+wiki15G_kd-code_64*150000"  # tiny6(fp16)+wiki(15G) kd代码预训练 64*150000
        # env['MODEL_PATH'] = "data/checkpoints/other/tiny6+wiki15G_kd-code_64*300000"  # tiny6(fp16)+wiki(15G) kd代码预训练 64*300000
        env['MODEL_ARGS'] = [
            '--block-lm', 
            '--num-layers 6', 
            '--hidden-size 768', 
            '--num-attention-heads 12', 
            '--max-position-embeddings 512', 
            '--tokenizer-model-type bert-base-uncased', 
            '--tokenizer-type BertWordPieceTokenizer', 
            '--load-pretrained ' + env['MODEL_PATH'],
            '--fp16',
            # '--fp32-allreduce',
        ]
        return env

class Models_pre:
    @staticmethod
    def block_tiny6(env: dict, **kw):
        env['gpt_options'] = [
            '--block-lm', 
            '--bert-prob 1.0', 
            '--experiment-name blocklm-blank', 
            '--num-layers 6', 
            '--hidden-size 768',
            '--num-attention-heads 12',
            '--seq-length 512',
            '--max-position-embeddings 512', 
            '--save data/checkpoints/pretrain/block_tiny6',  # 模型保存位置
            # '--load data/checkpoints/pretrain/block_tiny6/blocklm-blank07-31-07-36',  # 保存文件夹名会和这个一样
            '--resume-dataloader',
            '--train-data wiki',
            '--no-lazy-loader',
            '--tokenizer-type BertWordPieceTokenizer', 
            '--tokenizer-model-type bert-base-uncased', 
            '--split 949,50,1',
            '--distributed-backend nccl',
            '--lr-decay-style cosine',
            '--lr-decay-iters 120000',
            '--lr-decay-ratio 0.05',
            '--warmup .05',
            '--fp16',  # 用 ds 还需要设置 deepspeed_config 中的 fp16
            # '--fp32-allreduce',
        ]
        env['deepspeed_config'] = 'config/config_block_tiny6.json'  # 包含 batch-size/fp16 等
        return env

    @staticmethod
    def block_base(env: dict, **kw):
        env['gpt_options'] = [
            '--block-lm', 
            '--bert-prob 1.0', 
            '--experiment-name blocklm-blank', 
            '--num-layers 12', 
            '--hidden-size 768',
            '--num-attention-heads 12',
            '--seq-length 512',
            '--max-position-embeddings 512', 
            '--save data/checkpoints/pretrain/block_base',  # 模型保存位置
            # '--load data/checkpoints/pretrain/blocklm-base-blank',  # 续跑
            # '--load data/checkpoints/pretrain/block_base/blocklm-blank07-23-14-42',
            # '--load data/checkpoints/pretrain/block_base/test/deepspeed-gpu2mp2',
            # '--load data/checkpoints/pretrain/block_base/test/deepspeed-gpu2mp1',
            # '--load data/checkpoints/pretrain/block_base/test/deepspeed-gpu1mp1',
            # '--load data/checkpoints/pretrain/block_base/test/torch.distributed-gpu2mp2',
            # '--load data/checkpoints/pretrain/block_base/test/torch.distributed-gpu2mp1',
            # '--load data/checkpoints/pretrain/block_base/test/torch.distributed-gpu1mp1',
            # '--load data/checkpoints/pretrain/block_base/test/gpu1mp1',
            '--resume-dataloader',
            '--train-data bert-base',
            '--no-lazy-loader',
            '--tokenizer-type BertWordPieceTokenizer', 
            '--tokenizer-model-type bert-base-uncased', 
            '--split 949,50,1',
            '--distributed-backend nccl',
            '--lr-decay-style cosine',
            '--lr-decay-iters 120000',
            '--lr-decay-ratio 0.05',
            '--warmup .05',
            '--fp16',  # 用 ds 还需要设置 deepspeed_config 中的 fp16
            # '--fp32-allreduce',
        ]
        env['deepspeed_config'] = 'config/config_block_base.json'  # 包含 batch-size/fp16 等
        return env

class Tasks:
    EPOCH_SINGLE = ''  # 训练多少 epoch
    BATCH_SIZE = '16'

    @staticmethod
    def copa(env: dict, **kw):
        env['TASK_NAME'] = 'COPA'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '700'
        env['XXLARGE_EPOCH'] = '100'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1', 
            '--pattern-id 0',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000', 
            '--log-interval 20', 
            '--eval-interval 1000', 
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0 1)'
        env['PROMPT_IDS'] = '(1 2)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def rte(env: dict, **kw):
        env['TASK_NAME'] = 'RTE'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '450'
        env['XXLARGE_EPOCH'] = '50'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1', 
            '--pattern-id 0',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000', 
            '--log-interval 50', 
            '--eval-interval 10000000', 
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0 1 2 3)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def boolq(env: dict, **kw):
        env['TASK_NAME'] = 'BoolQ'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '120'
        env['XXLARGE_EPOCH'] = '24'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1', 
            '--pattern-id 4',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000', 
            '--log-interval 50', 
            '--eval-interval 10000000', 
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0 1 2 3 4 5)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def wic(env: dict, **kw):
        env['TASK_NAME'] = 'WiC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '220'
        env['XXLARGE_EPOCH'] = '40'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1', 
            '--pattern-id 1',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000', 
            '--log-interval 50', 
            '--eval-interval 10000000', 
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0 1 2)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def cb(env: dict, **kw):
        env['TASK_NAME'] = 'CB'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '500'
        env['XXLARGE_EPOCH'] = '100'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1', 
            '--pattern-id 3',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000', 
            '--log-interval 50', 
            '--eval-interval 10000000', 
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0 1 2 3)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def multirc(env: dict, **kw):
        env['TASK_NAME'] = 'MultiRC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '512'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '350'
        env['XXLARGE_EPOCH'] = '12'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1', 
            '--pattern-id 0',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000', 
            '--log-interval 50', 
            '--eval-interval 10000000', 
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0 1 2 3)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def wsc_generative(env: dict, **kw):
        env['TASK_NAME'] = 'WSC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}_generative'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '128'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '800'
        env['XXLARGE_EPOCH'] = '100'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000', 
            '--log-interval 50', 
            '--eval-interval 1000', 
            '--eval-iters 100',
        ]
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def wsc(env: dict, **kw):
        env['TASK_NAME'] = 'WSC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}-negative'
        env['MAX_SEQ_LEN'] = '128'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '500'
        env['XXLARGE_EPOCH'] = '100'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1',
            '--loss-func mix',
            '--wsc-negative',
            '--length-penalty 1',
            '--pattern-id 2',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000',
            '--log-interval 50',
            '--eval-interval 1000',
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0 1 2)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def record(env: dict, **kw):
        env['TASK_NAME'] = 'ReCoRD'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '512'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '50'
        env['XXLARGE_EPOCH'] = '3'
        env['TRAIN_ARGS'] = [
            '--lr-decay-style linear', 
            '--warmup 0.1', 
            '--weight-decay 1.0e-1',
            '--pattern-id 0',
        ]
        env['COMMON_ARGS'] = [
            '--save-interval 10000',
            '--log-interval 50',
            '--eval-interval 1000',
            '--eval-iters 100',
        ]
        env['PATTERN_IDS'] = '(0)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

class Scripts:
    @staticmethod
    def finetune_superglue(model_f, task_f, env={}, n_gpu=1, **kw):
        env['DATA_ROOT'] = 'data/english_data/superglue'  # 总数据位置
        model_f(env)
        task_f(env)
        env['SAVE_PATH'] = env['MODEL_PATH'] + '/finetune/' + env['TASK_NAME']
        env['N_GPU'] = f'{n_gpu}'  # BATCH_SIZE 均分到几张卡上
        env['PER_GPU_BS'] = str(int(int(env['BATCH_SIZE']) / int(env['N_GPU'])))
        env['EXPERIMENT_NAME'] = env['EXPERIMENT_NAME'] + '-' + datetime.now().strftime('%m-%d-%H-%M')
        py_args = [
            '--finetune',
            '--cloze-eval',
            '--experiment-name ' + env['EXPERIMENT_NAME'],
            '--task ' + env['TASK_NAME'],
            '--data-dir ' + env['DATA_PATH'],
            '--save ' + env['SAVE_PATH'],
            '--seq-length ' + env['MAX_SEQ_LEN'],
            '--checkpoint-activations',
            '--eval-batch-size 16',
            '--save-epoch 100000',
            *env['MODEL_ARGS'],
            *env['TRAIN_ARGS'],
            *env['COMMON_ARGS'],
            '--batch-size ' + env['PER_GPU_BS'],
            '--epochs ' + env['EPOCH_SINGLE'],
            '--lr ' + env['LR_SINGLE'],
            '--overwrite',
            '--num-workers 0',  # 不使用多进程数据加载器方便调试
        ]
        return py_args

    @staticmethod
    def pretrain_nvidia(model_pre_f, ds=True, env={}, **kw):
        model_pre_f(env)
        py_args = [
            *env['gpt_options'],
            '--checkpoint-activations',
            '--train-iters 123456789',  # 迭代几次
            '--model-parallel-size 1',  # 模型并行数, 常调参数
            # '--save-interval 100',  # 迭代几次保存一次, 默认 5000
        ]
        if ds:
            py_args += [
                '--deepspeed-activation-checkpointing',
                '--deepspeed',
                '--deepspeed_config ' + env['deepspeed_config'],
            ]
        return py_args


def split_py_args(py_args: list):  # 空格/等号切分成一节节, 路径不能有空格?
    args = []
    for py_arg in py_args:
        py_arg = py_arg.strip()
        if ' ' in py_arg or '=' in py_arg:
            args += re.split('[ =]+', py_arg)
        else:
            args.append(py_arg)
    return args


def create_cmd(script, model=None, model_pre=None, task=None, ds=False, gpus='6'):  # 生成可执行命令
    if ds:
        prefix = [
            'NCCL_DEBUG=info',
            'NCCL_IB_DISABLE=0',
            'NCCL_NET_GDR_LEVEL=2',
            'deepspeed',
            '--master_port=12367',
            f"--include=localhost:{gpus}",  # 占用显卡
            '--hostfile=',
        ]
    else:
        prefix = [
            f'CUDA_VISIBLE_DEVICES={gpus}',  # 占用显卡
            'python',
            '-u',
        ]
    n_gpu = gpus.count(',') + 1  # 不管预训练固定bs和模型并行问题
    py_args = split_py_args(script(model_f=model, task_f=task, model_pre_f=model_pre, ds=ds, n_gpu=n_gpu))
    py = f"{script.__name__.split('_')[0]}_glm.py"
    cmd = ' '.join(prefix + [py] + py_args)
    return cmd


if __name__ == '__main__':
    model = model_pre = task = None
    print()
    script = Scripts.finetune_superglue
    model = Models.block_tiny6
    task = Tasks.copa  # copa rte boolq wic cb multirc wsc_generative wsc record
    deepspeed = False
    print(create_cmd(script, model, model_pre, task, deepspeed))
    print()
    script = Scripts.pretrain_nvidia
    model_pre = Models_pre.block_tiny6
    deepspeed = True
    print(create_cmd(script, model, model_pre, task, deepspeed))
    print()