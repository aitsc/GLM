import sys, os
sys.path.append(os.getcwd())

from tsc.ab_配置生成器 import Scripts, Models_pre, Models, Tasks, create_cmd
import json
import os
from datetime import datetime
from pprint import pprint
from utils import ensure_directory_exists


if __name__ == "__main__":
    script = Scripts.finetune_superglue
    model = Models.block_tiny6
    gpus = sys.argv[1]  # 5

    max_output_L = []
    max_output_path = f"data/tmp_max_output/{datetime.now().strftime('%y%m%d_%H%M%S')}.json"
    ensure_directory_exists(max_output_path)
    for task in [
        Tasks.copa,
        Tasks.wsc_generative,
        Tasks.cb,
        Tasks.rte,
        Tasks.boolq,
        Tasks.wic,
        # Tasks.multirc,
        # Tasks.wsc,
        # Tasks.record,
    ]:
        custom_tmp_result = datetime.now().strftime('%y%m%d_%H%M%S')
        cmd = create_cmd(script, model=model, model_pre=None, task=task, ds=False, gpus=gpus)
        cmd += f' --custom_tmp_result data/tmp_result/{custom_tmp_result}.json'
        print(cmd, '\n')
        os.system(cmd)
        # 处理输出
        print(str(datetime.now()))
        with open(f'data/tmp_result/{custom_tmp_result}.json', 'r', encoding='utf8') as r:
            max_output_L.append(json.load(r))
        with open(max_output_path, 'w', encoding='utf8') as w:
            json.dump(max_output_L, w, ensure_ascii=False, indent=2)
        pprint(max_output_L)
        print()
