from dataclasses import dataclass
from typing import List, Optional, Any

import transformers
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import ShardedDDPOption, FSDPOption


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     """
#     Hack on the TrainingArguments class to avoid typing issues with
#     OmegaConf.
#     """

#     debug: List[DebugOption]
#     sharded_ddp: List[ShardedDDPOption]
#     fsdp: List[FSDPOption]
#     fsdp_config: dict

#     deepspeed_plugin: Optional[Any] = None