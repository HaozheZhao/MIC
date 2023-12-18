
import random
import logging
from transformers import Trainer
import matplotlib.pyplot as plt

import time
from typing import List, Optional, Union
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
import math

from torch.utils.tensorboard import SummaryWriter

from transformers.file_utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )

from typing import Dict, OrderedDict, Union, Any

from typing import Any, Dict, List, Optional, Tuple, Union, OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class InstructBLIP2Trainer(Trainer):

    def __init__(
        self,
        processor,
        config,
        model_args,
        predict_dataset=None,
        *args,
        **kwargs,
    ):
        
        super().__init__(*args, **kwargs)
        if 'args' in kwargs:
            self.training_args = kwargs['args']

        self.test_key = "accuracy"
        self.processor = processor
        self.model_args = model_args
        self.best_metrics = OrderedDict(
            {
                "best_epoch": 0,
                f"best_eval_{self.test_key}": 0,
            }
        )

        self.predict_dataset = predict_dataset
        self.epoch = 0

        self.config = config
        
        self.writer = SummaryWriter(
            f"./tensorboard_log/{self.model_args.experiment_name}"
        )
        self.model_args = model_args



        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        if 'test_dataset' in kwargs:
            self.test_dataset = kwargs['test_dataset']
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if hasattr(self.model_args,'special_visual_token_id'):
            inputs['sp_token'] = self.model_args.special_visual_token_id
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("min_length") is None and gen_kwargs.get("max_min_tokens") is None:
            gen_kwargs["min_length"] = self.args.generation_min_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("min_length") is None and gen_kwargs.get("max_min_tokens") is None:
            gen_kwargs["min_length"] = self.args.generation_min_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if hasattr(self.model_args,'special_visual_token_id'):
            gen_kwargs['sp_token'] = self.model_args.special_visual_token_id
        
        if 'qformer_input_ids' in inputs:
            generated_tokens = self.model.generate(
                pixel_values = inputs['pixel_values'],
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                img_mask = inputs['img_mask'],
                qformer_input_ids = inputs['qformer_input_ids'],
                qformer_attention_mask = inputs['qformer_attention_mask'],
                # **inputs,
                **gen_kwargs,
            )
        else:
            generated_tokens = self.model.generate(
                pixel_values = inputs['pixel_values'],
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                img_mask = inputs['img_mask'],
                # **inputs,
                **gen_kwargs,
            )

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        # in case the batch is shorter than max length, the output should be padded
        # if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
        #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        # elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
        #     gen_kwargs["max_new_tokens"] + 1
        # ):
        #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    if 'loss_input_ids'in inputs.keys(): 
                        outputs = model(
                        pixel_values = inputs['pixel_values'],
                        input_ids = inputs['loss_input_ids'],
                        attention_mask = inputs['loss_attention_mask'],
                        labels = inputs['loss_labels'],
                        img_mask = inputs['img_mask'],
                        **gen_kwargs
                    )
                    else:
                        outputs = model(**inputs)
                    
                    
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            # if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
            #     labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            # elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
            #     gen_kwargs["max_new_tokens"] + 1
            # ):
            #     labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if "train" in self.model_args.eval_type:
                logger.info(f"***** Running Evaluation for train dataset *****")
                metrics = self.evaluate(
                        eval_dataset=self.train_dataset,
                        ignore_keys=ignore_keys_for_eval,
                    )
                self._report_to_hp_search(trial, epoch, metrics)
            # if "test" in self.model_args.eval_type:
            #     logger.info(f"***** Running Evaluation for test dataset *****")
            #     metrics = self.evaluate(ignore_keys=ignore_keys_for_eval, eval_dataset=self.test_dataset)
            #     self._report_to_hp_search(trial, epoch, metrics)
            # else:
            logger.info(f"***** Running Evaluation for eval dataset *****")    
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]
                # self.best_model = model
                # self._save_checkpoint(self.best_model , trial, metrics=eval_metrics)

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)