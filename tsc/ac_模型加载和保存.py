import sys, os
sys.path.append(os.getcwd())

from model import GLMModel
from model import GLMForMultiTokenCloze, GLMForMultiTokenClozeFast, GLMForSingleTokenCloze, GLMForSequenceClassification
import torch
from model.modeling_bert import BertForMultipleChoice, BertForSequenceClassification
from tasks.superglue.dataset import PROCESSORS
from tasks.superglue.pvp import PVPS
from tasks.superglue.dataset import MULTI_CHOICE_DATASETS
from arguments import get_args
import random
import numpy as np
from model import PyTorchDistributedDataParallel as TorchDDP, DistributedDataParallel as LocalDDP
import mpu
from fp16 import FP16_Module, FP16_Optimizer, DynamicLossScaler
from utils import print_rank_0, get_checkpoint_name, get_checkpoint_iteration
from filelock import FileLock
import pathlib
from configure_data import prepare_tokenizer
import pretrain_glm


def get_model(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Build the model."""
    print_rank_0('building GPT2 model ...')
    if args.pretrained_bert:
        if model_type == "multiple_choice":
            model = BertForMultipleChoice.from_pretrained(args.tokenizer_model_type,
                                                          cache_dir=args.cache_dir,
                                                          fp32_layernorm=args.fp32_layernorm,
                                                          fp32_embedding=args.fp32_embedding,
                                                          layernorm_epsilon=args.layernorm_epsilon)
        elif model_type == "classification":
            model = BertForSequenceClassification.from_pretrained(args.tokenizer_model_type,
                                                                  cache_dir=args.cache_dir,
                                                                  fp32_layernorm=args.fp32_layernorm,
                                                                  fp32_embedding=args.fp32_embedding,
                                                                  layernorm_epsilon=args.layernorm_epsilon,
                                                                  num_labels=num_labels)
        else:
            raise NotImplementedError
    else:
        output_predict, paralle_output = True, True
        if (model_type == "multiple_choice" or model_type == "classification") and not args.cloze_eval:
            output_predict = False
        if model_type is not None:
            paralle_output = False
        if spell_length is not None:
            print_rank_0(f"Continuous spell length {spell_length}")
        model = GLMModel(num_layers=args.num_layers,
                         vocab_size=args.vocab_size,
                         hidden_size=args.hidden_size,
                         num_attention_heads=args.num_attention_heads,
                         embedding_dropout_prob=args.hidden_dropout,
                         attention_dropout_prob=args.attention_dropout,
                         output_dropout_prob=args.hidden_dropout,
                         max_sequence_length=args.max_position_embeddings,
                         max_memory_length=args.mem_length,
                         checkpoint_activations=args.checkpoint_activations,
                         checkpoint_num_layers=args.checkpoint_num_layers,
                         parallel_output=paralle_output,
                         relative_encoding=args.transformer_xl,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=output_predict,
                         spell_length=spell_length,
                         spell_func=args.prompt_func,
                         attention_scale=args.attention_scale)
        if args.freeze_transformer:
            model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
        if model_type is not None:
            if model_type == 'multiple_choice':
                if args.cloze_eval:
                    if multi_token:
                        if args.fast_decode:
                            model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
                        else:
                            model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
                    else:
                        model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
                else:
                    model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                         num_class=num_labels)
            elif model_type == 'classification':
                model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                     num_class=num_labels)
            elif model_type == 'generation':
                pass
            else:
                raise NotImplementedError(model_type)
    print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)  # 统计参数总量
    # GPU allocation.
    model.cuda(torch.cuda.current_device())
    model = LocalDDP(model)
    return model

def load_pretrained(model, checkpoint_path, args, task_tokens=None):
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    if mpu.get_data_parallel_rank() == 0:
        print_rank_0('global rank {} is loading pretrained model {}'.format(
            torch.distributed.get_rank(), checkpoint_name))
    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')
    if args.deepspeed:
        model = model.module
    if isinstance(model, TorchDDP):
        model = model.module
    if isinstance(model, FP16_Module):
        model = model.module
    if hasattr(model, "model"):
        model = model.model

    # Model.
    def extend_embedding_weights(state_weights, model_weights):
        original_length = state_weights.shape[0]
        assert original_length <= args.max_position_embeddings + 1
        new_weights = model_weights.clone()
        new_weights[:original_length] = state_weights
        return new_weights

    if args.block_lm:
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            position_weights = sd['module']["transformer.position_embeddings.weight"]
            if args.max_position_embeddings + 1 > position_weights.shape[0]:
                sd['module']["transformer.position_embeddings.weight"] = extend_embedding_weights(
                    position_weights, model.state_dict()["transformer.position_embeddings.weight"].data)
                print_rank_0(f"Extend position embedding to {args.max_position_embeddings + 1}")
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            block_position_weights = sd['module']["transformer.block_position_embeddings.weight"]
            if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
                sd['module']["transformer.block_position_embeddings.weight"] = extend_embedding_weights(
                    block_position_weights,
                    model.state_dict()["transformer.block_position_embeddings.weight"].data)
                print_rank_0(f"Extend block position embedding to {args.max_position_embeddings + 1}")
    missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
    if args.continuous_prompt and args.prompt_init:
        model.prompt_spell.init_embedding(model.word_embeddings.weight.data, task_tokens)


def main():
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # 参数
    args = get_args()
    # mpu 依赖初始化的 torch.distributed
    args.master_ip = os.getenv('MASTER_ADDR', 'localhost')
    args.master_port = os.getenv('MASTER_PORT', '6000')
    init_method = 'tcp://' + args.master_ip + ':' + args.master_port
    torch.distributed.init_process_group(
        backend='nccl',  # 指定当前进程要使用的通信后端
        world_size=1,  # 该 job 中的总进程数。如果指定 store 参数，则需要指定该参数
        rank=0,  # 表示当前进程的编号，即优先级。如果指定 store 参数，则必须指定该参数. rank=0 的为主进程，即 master 节点
        init_method=init_method,  # 指定当前进程组初始化方式。如果未指定 init_method 及 store，则默认为 env://，表示使用读取环境变量的方式进行初始化。该参数与 store 互斥
    )
    # 初始化 mpu
    mpu.initialize_model_parallel(args.model_parallel_size)
    # 初始化种子
    seed = args.seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # 获取不同需求的模型初始化参数
    superglue_tasks = list(PROCESSORS.keys())
    model_kwargs = {}
    if args.task.lower() in superglue_tasks:
        processor = PROCESSORS[args.task.lower()](args)
        pvp = PVPS[args.task.lower()](args, None, processor.get_labels(), args.seq_length,
                                    pattern_id=args.pattern_id, is_multi_token=args.multi_token,
                                    num_prompt_tokens=args.num_prompt_tokens)
        if args.continuous_prompt:
            model_kwargs["spell_length"] = pvp.spell_length
        else:
            if args.cloze_eval:
                multi_token = pvp.is_multi_token
            else:
                multi_token = args.task.lower() in MULTI_CHOICE_DATASETS
            args.multi_token = multi_token
            if not multi_token:
                model_kwargs["model_type"] = "multiple_choice" if args.cloze_eval else "classification"
                model_kwargs["multi_token"] = False
                model_kwargs["num_labels"] = len(processor.get_labels())
            else:
                model_kwargs["model_type"] = "multiple_choice"
                model_kwargs["multi_token"] = True
                model_kwargs["num_labels"] = 1

    # 构建模型结构
    tokenizer = prepare_tokenizer(args)
    pretrain_glm.tokenizer = tokenizer
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    model = get_model(args, **model_kwargs)
    # 加载模型参数
    with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
        load_pretrained(model, args.load_pretrained, args)


if __name__ == '__main__':
    main()