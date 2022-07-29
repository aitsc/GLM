import data_utils
import blocklm_utils
from tasks.superglue import dataset as tasks_superglue_dataset
from torch.utils import data as torch_utils_data
from tasks.superglue import pvp as tasks_superglue_pvp
from tasks import data_utils as tasks_data_utils


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
        data_utils.samplers.RandomSampler, torch_utils_data.SequentialSampler
        data_utils.samplers.DistributedBatchSampler, torch_utils_data.BatchSampler
        blocklm_utils.ConstructBlockStrategy.construct_blocks
        torch_utils_data.DataLoader

    def 微调():
        data_utils.tokenization.BertWordPieceTokenizer
        tasks_superglue_pvp.RtePVP
        tasks_superglue_dataset.RteProcessor
        tasks_superglue_dataset.SuperGlueDataset
        torch_utils_data.DistributedSampler
        tasks_data_utils.my_collate
        torch_utils_data.DataLoader
        tasks_data_utils.build_input_from_ids