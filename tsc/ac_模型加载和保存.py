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


def get_model(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    """Build the model."""
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
            print(f"Continuous spell length {spell_length}")
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
    print(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])), flush=True)  # 统计参数总量
    # GPU allocation.
    model.cuda(torch.cuda.current_device())
    return model


def main():
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    seed = args.seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    superglue_tasks = list(PROCESSORS.keys())
    if args.task.lower() in superglue_tasks:
        model_kwargs = {}
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
        model = get_model(args, **model_kwargs)
    else:
        model = get_model(args)



if __name__ == '__main__':
    main()