from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from tasks.utils import *


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # NOTE 没用到
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    # NOTE 没用到
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    # NOTE 没用到
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default='dialog_version_control/data/ATIS/train.json', metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default='dialog_version_control/data/ATIS/test.json',
        metadata={"help": "A csv or a json file containing the test data."}
    )
    label_file: Optional[str] = field(
        default='dialog_version_control/data/ATIS/label.txt',
        metadata={"help": "A txt file containing the label data."}
    )
    dev_rate: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "For spliting a dev set"
        },
    )
    use_preprocessed: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use preprocessed data"
        },
    )
    do_full_training: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to using all data"
        },
    )
    only_evaluate: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to only test the result"
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # NOTE 没用到
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # NOTE 没用到
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # NOTE 没用到
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # NOTE 没用到
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # NOTE 没用到
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    pre_seq_len: int = field(
        default=6,
        metadata={
            "help": "The length of prompt"
        }
    ) # 可能会引起误会，datasets内也定义了pre_seq_len
    task_type: Optional[str] = field(
        default="language_modeling",
        metadata={
            "help": "Design which head to use."
        }
    )
    eval_type: Optional[str] = field(
        default="eval",
        metadata={
            "help": "Design which head to use."
        }
    )
    prompt_type: Optional[str] = field(
        default="soft",
        metadata={
            "help": "Use hard or soft prompt"
        }
    )
    template_id: Optional[str] = field(
        default="template_0",
        metadata={
            "help": "The specific soft prompt template to use"
        }
    )
    verbalizer_id: Optional[str] = field(
        default="verbalizer_0",
        metadata={
            "help": "The specific verbalizer to use"
        }
    )
    prompt_operation: Optional[str] = field(
        default="mean",
        metadata={
            "help": "Will use max, sum, mean, attention or cross-attention soft prompt tuning during training"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    num_attention_layers: int = field(
        default=1,
        metadata={
            "help": ""
        }
    )
    num_attention_heads: int = field(
        default=8,
        metadata={
            "help": ""
        }
    )
    whether_PositionalEncoding: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    whether_PositionalWiseFeedForward: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    fix_deberta: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    data_augmentation: Optional[str] = field(
        default="none",
        metadata={
            "help": "rdrop, AT, mixup, manifold_mixup"
        }
    )
    model_type: Optional[str] = field(
        default="blip2",
        metadata={
            "help": "blip2, instructblip"
        }
    )
    label: Optional[str] = field(
        default="label",
        metadata={
            "help": ""
        }
    )
    experiment_name: Optional[str] = field(
        default="label",
        metadata={
            "help": ""
        }
    )
# Negative Sample
    negative_sample_num: Optional[int] = field(
        default=1,
        metadata={
            "help": ""
        }

    )
    processor_path: Optional[str] = field(
        default=None,
        metadata={
            "help": ""
        }

    )


@dataclass
class ExtraTrainingArguments(TrainingArguments):
    generation_max_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "generation_max_length"
        }
    )
    generation_min_length: Optional[int] = field(
        default=1,
        metadata={
            "help": "generation_min_length"
        }
    )
    generation_num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "generation_num_beams"
        }
    )
    predict_with_generate: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
    multiple_choice : bool = field(
        default=False,
        metadata={
            "help": ""
        }
    )
    few_shot : bool = field(
        default=False,
        metadata={
            "help": ""
        }
    )
    using_instruct_qformer: bool = field(
        default=True,
        metadata={
            "help": ""
        }
    )
def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtraTrainingArguments))

    args = parser.parse_args_into_dataclasses()

    return args