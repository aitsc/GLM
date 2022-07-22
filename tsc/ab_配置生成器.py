


model_args = [
    '--block-lm', 
    '--num-layers 12', 
    '--hidden-size 768', 
    '--num-attention-heads 12', 
    '--max-position-embeddings 512', 
    '--tokenizer-model-type bert-base-uncased', 
    '--tokenizer-type BertWordPieceTokenizer', 
    '--load-pretrained data/checkpoints/blocklm-base-blank',
]

train_args = [
    '--lr-decay-style linear', 
    '--warmup 0.1', 
    '--weight-decay 1.0e-1', 
    '--pattern-id 0',
]

common_args = [
    '--save-interval 10000', 
    '--log-interval 20', 
    '--eval-interval 1000', 
    '--eval-iters 100',
]

main_args = [
    '--local_rank=0', 
    '--finetune', 
    '--cloze-eval', 
    '--experiment-name blank-base-copa-07-22-04-26', 
    '--task COPA', 
    '--data-dir data/english_data/superglue/COPA', 
    '--save data/finetune_checkpoints/COPA', 
    '--seq-length 256', 
    '--checkpoint-activations', 
    '--eval-batch-size 16', 
    '--save-epoch 100000', 
    '--fp16', 
    '--batch-size 16', 
    '--epochs 5000', 
    '--lr 1e-5', 
    '--overwrite',
]