import data_utils
import blocklm_utils
import torch


class 数据集:
    def 预训练():
        data_utils.corpora.BertBaseData.PATH  # 数据路径
        data_utils.corpora.DataReader.process  # 数据主处理
        data_utils.corpora.PromptReader.tokenize_worker  # 包含json处理
        data_utils.corpora.BertData.process_line  # prompt, text
        data_utils.LazyWriter
        data_utils.LazyLoader
        data_utils.corpora.PromptDataset
        data_utils.datasets.ConcatDataset
        data_utils.datasets.SplitDataset
        data_utils.datasets.BlockDataset
        data_utils.samplers.RandomSampler, torch.utils.data.SequentialSampler
        data_utils.samplers.DistributedBatchSampler, torch.utils.data.BatchSampler
        blocklm_utils.ConstructBlockStrategy.construct_blocks
        torch.utils.data.DataLoader