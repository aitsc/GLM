from datetime import datetime
import re

class Models:
    @staticmethod
    def model_blocklm_base(env: dict):
        env['MODEL_TYPE'] = "blank-base"
        env['MODEL_ARGS'] = [
            '--block-lm', 
            '--num-layers 12', 
            '--hidden-size 768', 
            '--num-attention-heads 12', 
            '--max-position-embeddings 512', 
            '--tokenizer-model-type bert-base-uncased', 
            '--tokenizer-type BertWordPieceTokenizer', 
            f'--load-pretrained {env["CHECKPOINT_PATH"]}/blocklm-base-blank',
        ]
        return env

class Tasks:
    EPOCH_SINGLE = '10000'  # 训练多少 epoch

    @staticmethod
    def task_copa(env: dict):
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-copa'
        env['TASK_NAME'] = 'COPA'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/COPA'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '50'
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
        env['BATCH_SIZE'] = '16'
        return env

    @staticmethod
    def task_rte(env: dict):
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-rte'
        env['TASK_NAME'] = 'RTE'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/RTE'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE if Tasks.EPOCH_SINGLE else '50'
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
        env['BATCH_SIZE'] = '16'
        return env

class Scripts:
    @staticmethod
    def finetune_superglue(model_f, task_f, env={}):
        env['DATA_ROOT'] = 'data/english_data/superglue'  # 数据位置
        env['CHECKPOINT_PATH'] = 'data/checkpoints'  # 模型位置
        model_f(env)
        task_f(env)
        env['SAVE_PATH'] = 'data/finetune_checkpoints/' + env['TASK_NAME']
        env['N_GPU'] = '1'
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
            '--fp16',
            '--batch-size ' + env['PER_GPU_BS'],
            '--epochs ' + env['EPOCH_SINGLE'],
            '--lr ' + env['LR_SINGLE'],
            '--overwrite',
        ]
        return py_args


def split_py_args(py_args: list):
    args = []
    for py_arg in py_args:
        py_arg = py_arg.strip()
        if ' ' in py_arg or '=' in py_arg:
            args += re.split('[ =]+', py_arg)
        else:
            args.append(py_arg)
    return args


if __name__ == '__main__':
    py_args = Scripts.finetune_superglue(Models.model_blocklm_base, Tasks.task_copa)
    print(split_py_args(py_args))