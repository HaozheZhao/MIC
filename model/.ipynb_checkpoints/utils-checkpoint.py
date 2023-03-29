from enum import Enum

from model.token_classification import (
    BertPrefixForTokenClassification,
    RobertaPrefixForTokenClassification,
    DebertaPrefixForTokenClassification,
    DebertaV2PrefixForTokenClassification
)

from model.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    DebertaPrefixForSequenceClassification
)

from model.question_answering import (
    BertPrefixForQuestionAnswering,
    RobertaPrefixModelForQuestionAnswering,
    DebertaPrefixModelForQuestionAnswering
)

from model.multiple_choice import (
    BertPrefixForMultipleChoice,
    RobertaPrefixForMultipleChoice,
    DebertaPrefixForMultipleChoice,
    BertPromptForMultipleChoice,
    RobertaPromptForMultipleChoice
)

from model.language_modeling import (
    BertForMaskedLM,
    BertPromptForMaskedLM,
    BertPrefixForMaskedLM
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoModelForMaskedLM
)

class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4,
    LANGUAGE_MODELING = 5
    
tasktype = {
    "language_modeling": TaskType.LANGUAGE_MODELING,
    "sequence_classification": TaskType.SEQUENCE_CLASSIFICATION,
    "multiple_choice": TaskType.MULTIPLE_CHOICE
}

PREFIX_MODELS = {
    "bert": {
        TaskType.TOKEN_CLASSIFICATION: BertPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: BertPrefixForMultipleChoice,
        TaskType.LANGUAGE_MODELING: BertPrefixForMaskedLM
    },
    "roberta": {
        TaskType.TOKEN_CLASSIFICATION: RobertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: RobertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
    },
    "deberta": {
        # TaskType.TOKEN_CLASSIFICATION: DebertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: DebertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: DebertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: DebertaPrefixForMultipleChoice,
    },
    "deberta-v2": {
        # TaskType.TOKEN_CLASSIFICATION: DebertaV2PrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: None,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    }
}

PROMPT_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: BertPromptForMultipleChoice,
        TaskType.LANGUAGE_MODELING: BertPromptForMaskedLM
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPromptForMultipleChoice
    }
}

AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
    TaskType.LANGUAGE_MODELING: AutoModelForMaskedLM
}

def get_model(model_args, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        model_class = PREFIX_MODELS[config.model_type][tasktype[model_args.task_type]]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            model_args=model_args,
            revision=model_args.model_revision,
        )
    elif model_args.prompt:
        model_class = PROMPT_MODELS[config.model_type][tasktype[model_args.task_type]]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            model_args=model_args,
            revision=model_args.model_revision,
        )
    else:
        # 直接fine-tune
        model_class = AUTO_MODELS[model_args.task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model


def get_model_deprecated(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        if task_type == TaskType.TOKEN_CLASSIFICATION:
            from model.token_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            from model.sequence_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.QUESTION_ANSWERING:
            from model.question_answering import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.MULTIPLE_CHOICE:
            from model.multiple_choice import BertPrefixModel

        if config.model_type == "bert":
            model = BertPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta":
            model = DebertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta-v2":
            model = DebertaV2PrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError


    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len

        from model.sequence_classification import BertPromptModel, RobertaPromptModel
        if config.model_type == "bert":
            model = BertPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError
            

    else:
        if task_type == TaskType.TOKEN_CLASSIFICATION:
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
            
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )

        elif task_type == TaskType.QUESTION_ANSWERING:
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif task_type == TaskType.MULTIPLE_CHOICE:
            model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
    
        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model
