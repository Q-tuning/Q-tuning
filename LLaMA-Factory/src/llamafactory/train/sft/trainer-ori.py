# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader
from typing_extensions import override
from .scorer import Llama_Scorer
from .custom_batchsampler import CustomBatchSampler

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

from .MMDLoss import mmd_kl_divergence

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import DataArguments, FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        reference_model, 
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        if reference_model != None:
            self.reference_model = reference_model.to(dtype=next(self.model.parameters()).dtype, device=self.model.device)
            self.reference_model.resize_token_embeddings(len(self.processing_class))
            self.reference_model.config.vocab_size = len(self.processing_class)
        self.token_method = data_args.token_method
        self.token_ratio = data_args.token_ratio
        # also store data_ratio for subset size in self-distill
        self.data_ratio = getattr(data_args, "data_ratio", 1.0)
        self.plug = getattr(data_args, "plug", None)
        # one-time print flag for budget plug after epoch > 0
        self._budget_logged_after_epoch0 = False
        # self-distillation plugin params
        self.self_distill = getattr(data_args, "self_distill", False)
        self.distill_temperature = getattr(data_args, "distill_temperature", 4.0)
        self.distill_lambda_1 = getattr(data_args, "distill_lambda_1", 1.0)
        self.distill_lambda_2 = getattr(data_args, "distill_lambda_2", 1.0)
        self.last_attn = None
        self._attention_hook_handle = None
        # self.complexity_scorer = Llama_Scorer(self.reference_model, self.tokenizer)
        # self.quality_scorer = Llama_Scorer(self.reference_model, self.tokenizer)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    # @override
    # def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
    #     if self.finetuning_args.disable_shuffling:
    #         # return torch.utils.data.SequentialSampler(self.train_dataset)
    #         return self.train_dataset.pruning_sampler()
    #     return self.train_dataset.pruning_sampler()

    #     return super()._get_train_sampler()

    def get_train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_sampler=CustomBatchSampler(self.train_dataset.pruning_sampler(), self.args.per_device_train_batch_size, False), 
            collate_fn=self.data_collator
        )
        return train_dataloader

    # @override
    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     return super().compute_loss(model, inputs, *args, **kwargs)
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1:
            self.last_attn = output[1]
        else:
            self.last_attn = output

    def _resolve_last_attention_module(self, model: "torch.nn.Module"):
        """
        Try to locate the last decoder layer's attention module in a robust way, compatible with
        PEFT-wrapped models and different backbone conventions (e.g., Qwen/LLaMA using `model.layers`,
        GPT-style using `transformer.h`). Returns the attention module or None if not found.
        """
        try:
            core_model = model.module if hasattr(model, "module") else model
            base = getattr(core_model, "get_base_model", lambda: core_model)()
            backbone = (
                getattr(base, "model", None)
                or getattr(base, "transformer", None)
                or getattr(base, "base_model", None)
            )
            if backbone is None:
                return None

            layers = getattr(backbone, "layers", None) or getattr(backbone, "h", None)
            if layers is None or len(layers) == 0:
                return None

            last_layer = layers[-1]
            attn = (
                getattr(last_layer, "self_attn", None)
                or getattr(last_layer, "attention", None)
                or getattr(last_layer, "attn", None)
            )
            return attn
        except Exception:
            return None

    @torch.no_grad()
    def _compute_ppl_entropy(self, logits: "torch.Tensor", shifted_labels: "torch.Tensor", attn_mask_shift: "torch.Tensor") -> tuple["np.ndarray", "np.ndarray"]:
        """
        Compute per-sample ppl and entropy with masks.
        - logits: [B, L-1, V]
        - shifted_labels: [B, L-1]
        - attn_mask_shift: [B, L-1] (1 for valid positions)
        """
        B, T, V = logits.shape
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        # token-level NLL (ignore positions where label == -100)
        nll = torch.zeros((B, T), device=logits.device, dtype=logits.dtype)
        valid_label_mask = (shifted_labels != -100) & (attn_mask_shift > 0)
        if valid_label_mask.any():
            nll[valid_label_mask] = -log_probs[valid_label_mask, shifted_labels[valid_label_mask]]
        # per-sample loss and ppl
        token_counts = valid_label_mask.sum(dim=1).clamp_min(1)
        loss_per_sample = (nll.sum(dim=1) / token_counts).to(dtype=torch.float32)
        ppl = torch.exp(loss_per_sample).detach().cpu().numpy()

        # token entropy H(p) = -sum p log p, averaged over valid attention positions
        token_entropy = (-probs * log_probs).sum(dim=-1)  # [B, T]
        valid_attn_mask = (attn_mask_shift > 0)
        ent_counts = valid_attn_mask.sum(dim=1).clamp_min(1)
        entropy_per_sample = (token_entropy * valid_attn_mask).sum(dim=1) / ent_counts
        entropy = entropy_per_sample.detach().cpu().numpy()
        return ppl, entropy

    def _dynamic_threshold_search_bisect(self, ppl: "np.ndarray", entropy: "np.ndarray", keep_ratio: float, tol: float = 0.01, max_iter: int = 20):
        """
        改进版：用分位数对称阈值，通过 α∈[0,0.49] 的二分搜索控制 Q2/Q4 规模，
        - Q2: ppl > Q_{1-α}(ppl) 且 entropy <= Q_{α}(entropy)
        - Q4: ppl <= Q_{α}(ppl) 且 entropy > Q_{1-α}(entropy)
        使 keep_pre ≈ keep_ratio，更稳健。
        返回 (ppl_thresh_mid, ent_thresh_mid, Q2_mask, Q4_mask)，其中阈值用于日志展示（取中值）。
        """
        import numpy as _np

        N = len(ppl)
        if N == 0:
            return None, None, _np.zeros((0,), dtype=bool), _np.zeros((0,), dtype=bool)

        # 边界保护
        def _quantile(x: _np.ndarray, q: float) -> float:
            q = float(min(max(q, 0.0), 1.0))
            return float(_np.quantile(x, q))

        # 二分 α 控制规模（α 越大，阈值越宽松，保留越多）
        alpha_low, alpha_high = 0.0, 0.49
        best = None

        for _ in range(max_iter):
            alpha = (alpha_low + alpha_high) / 2.0
            ppl_hi = _quantile(ppl, 1.0 - alpha)
            ppl_lo = _quantile(ppl, alpha)
            ent_lo = _quantile(entropy, alpha)
            ent_hi = _quantile(entropy, 1.0 - alpha)

            Q2 = (ppl > ppl_hi) & (entropy <= ent_lo)
            Q4 = (ppl <= ppl_lo) & (entropy > ent_hi)
            keep_actual = (Q2.sum() + Q4.sum()) / max(1, N)

            best = (alpha, ppl_hi, ppl_lo, ent_lo, ent_hi, Q2, Q4)
            if abs(keep_actual - keep_ratio) < tol:
                break
            if keep_actual < keep_ratio:
                # 保留太少 → 增大 α（放宽阈值）
                alpha_low = alpha
            else:
                # 保留太多 → 减小 α（收紧阈值）
                alpha_high = alpha

        # 输出中位阈值用于日志（仅展示，无严格语义）
        alpha, ppl_hi, ppl_lo, ent_lo, ent_hi, Q2, Q4 = best
        ppl_mid = (ppl_lo + ppl_hi) / 2.0
        ent_mid = (ent_lo + ent_hi) / 2.0

        # 附加：记录 α 到 self.log（若可用）
        try:
            if getattr(self, "state", None) is not None and self.state.global_step % max(1, int(getattr(self.args, "logging_steps", 10) or 10)) == 0:
                self.log({
                    "wisely/alpha": float(alpha),
                    "wisely/keep_actual": float((Q2.sum() + Q4.sum()) / max(1, N)),
                })
        except Exception:
            pass

        return ppl_mid, ent_mid, Q2, Q4

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        
        # Move inputs to the correct device
        inputs = {k: v if k == "text" else v.to(model.device) for k, v in inputs.items()}

        indices = inputs['indices']
        weights = inputs['weights']
        self.indices = indices

        shifted_labels = inputs['labels'][:, 1:].contiguous().detach()
        attention_mask = inputs['attention_mask'][:, 1:].contiguous().detach()
        
        # Compute reference loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')#Equation One
        
        # Forward pass with target model
        outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels'],
                return_dict=True, 
                output_hidden_states = True,
                output_attentions = (self.token_method in ["sample_fastv", "sample_sparsevlm"]), # can output attention weights but will slow down training
            )
        target_logits = outputs.logits[:, :-1, :]
        target_token_loss = loss_fct(#Equation Two
            target_logits.reshape(-1, target_logits.size(-1)),
            shifted_labels.reshape(-1)
        ).reshape(shifted_labels.size())

        # Apply attention mask
        target_token_loss = target_token_loss * attention_mask
        if self.train_dataset.method == "infobatch":
            self.train_dataset.__setscore__(indices.detach().cpu().numpy(), target_token_loss.mean(dim=1).detach().cpu().float().numpy())

        # if self.train_dataset.method == "deita" or self.train_dataset.method == "deita-grand" or self.train_dataset.method == "deita-el2n":
        if "deita" in self.train_dataset.method:
            # auto_scorer = Llama_Scorer(model, self.processing_class)
            # complexities = []
            # qualities = []
            # for item in inputs['text']:
            #     complexity = auto_scorer.infer_complexity(item["instruction"])
            #     quality = auto_scorer.infer_quality(item["instruction"], item["output"])
            #     complexities.append(complexity)
            #     qualities.append(quality)
            # self.train_dataset.__setcomplexity__(indices.detach().cpu().numpy(), np.array(complexities, dtype=np.float32))
            # self.train_dataset.__setquality__(indices.detach().cpu().numpy(), np.array(qualities, dtype=np.float32))

            self.train_dataset.__setmodel__(model, self.processing_class)

            seq_len = attention_mask.sum(1, keepdim = True)
            if self.processing_class.padding_side == "right":
                last_hidden_state = outputs.hidden_states[-1][torch.arange(seq_len.size(0))[:, None], seq_len - 1]
            elif self.processing_class.padding_side == "left":    
                last_hidden_state = outputs.hidden_states[-1][:, -1]
            self.train_dataset.__setembed__(indices.detach().cpu().numpy(), last_hidden_state.detach().cpu().float().numpy())
        if self.train_dataset.method == "deita-el2n":
            vocab_size = self.model.config.vocab_size  
            labels = inputs['labels'].clone()
            labels[labels == -100] = self.processing_class.pad_token_id
            all_logits = outputs.logits.detach().to(dtype=torch.float16, device=model.device)
            probabilities = torch.nn.functional.softmax(all_logits, dim=-1, dtype=torch.float16).to(model.device)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=vocab_size).half().to(model.device)
            valid_len = min(probabilities.size(1), labels.size(1))  
            labels_one_hot = labels_one_hot[:, :valid_len, :]
            probabilities = probabilities[:, :valid_len, :]
            l2_norm = torch.norm(probabilities - labels_one_hot, dim=-1, dtype=torch.float16).to(model.device)
            mean_l2_norm = l2_norm.sum(dim=1, dtype=torch.float16) / valid_len
            self.train_dataset.__setel2n__(indices.detach().cpu().numpy(), mean_l2_norm.detach().cpu().float().numpy())

        # Build sub-batch B' indices according to data_ratio for both branches
        batch_size = inputs['input_ids'].size(0)
        keep_ratio = float(self.train_dataset.ratio) if hasattr(self.train_dataset, "ratio") else float(getattr(self, "data_ratio", 1.0))
        subset_size = max(1, int(round(batch_size * keep_ratio)))
        perm = torch.randperm(batch_size, device=model.device)
        idx_sub = perm[:subset_size]

        # Plugin: budget (sample-level selection first by data_ratio, then global token budget within selected samples)
        if getattr(self, "plug", None) == "budget":
            # Build global valid mask over the whole batch (exclude special tokens)
            shifted_labels_b = shifted_labels  # [B, L-1]
            attn_mask_shift_b = attention_mask  # [B, L-1]
            allowed_mask_b = (attn_mask_shift_b > 0)
            try:
                special_ids = getattr(self.processing_class, "all_special_ids", None)
            except Exception:
                special_ids = None
            if special_ids is not None and isinstance(special_ids, (list, tuple)) and len(special_ids) > 0:
                special_mask = torch.zeros_like(allowed_mask_b, dtype=torch.bool)
                cur_tokens = inputs['input_ids'][:, 1:]
                for sid in special_ids:
                    special_mask |= (cur_tokens == sid)
                allowed_mask_b = allowed_mask_b & (~special_mask)

            # Compute global totals over full batch for budget, but only select tokens within selected samples
            total_valid_all = int(allowed_mask_b.sum().item())
            budget_ratio = float(self.data_ratio) * float(self.token_ratio)
            n_budget = max(1, int(round(total_valid_all * budget_ratio)))

            # Limit selection to the sample subset idx_sub (data pruning stage)
            allowed_mask_sel = allowed_mask_b[idx_sub]
            total_valid_sel = int(allowed_mask_sel.sum().item())
            n_budget = min(n_budget, max(1, total_valid_sel))

            # Score tokens according to token_method within selected samples only
            if self.token_method == "random":
                scores_sel = torch.rand_like(allowed_mask_sel.reshape(-1), dtype=target_token_loss.dtype)
                scores_sel[~allowed_mask_sel.reshape(-1)] = -1e9
            elif self.token_method == "rho" and hasattr(self, "reference_model") and self.reference_model is not None:
                with torch.no_grad():
                    ref_outputs = self.reference_model(
                        input_ids=inputs['input_ids'][idx_sub],
                        attention_mask=inputs['attention_mask'][idx_sub],
                        labels=inputs['labels'][idx_sub],
                        return_dict=True
                    )
                    ref_logits = ref_outputs.logits[:, :-1, :]
                    ref_token_loss_sel = loss_fct(
                        ref_logits.reshape(-1, ref_logits.size(-1)),
                        shifted_labels[idx_sub].reshape(-1)
                    ).reshape(shifted_labels[idx_sub].size())
                    ref_token_loss_sel = (ref_token_loss_sel * attention_mask[idx_sub]).to(dtype=target_token_loss.dtype)
                excess_sel = (target_token_loss[idx_sub] - ref_token_loss_sel).reshape(-1)
                scores_sel = excess_sel
                scores_sel[~allowed_mask_sel.reshape(-1)] = -1e9
            elif self.token_method == "wise":
                # ppl(i) + neighbors within allowed positions
                valid = allowed_mask_sel.to(dtype=target_token_loss.dtype)
                per_token_loss = target_token_loss[idx_sub]
                ppl = torch.exp(per_token_loss) * valid
                left = F.pad(ppl[:, :-1], (1, 0)) * F.pad(valid[:, :-1], (1, 0))
                right = F.pad(ppl[:, 1:], (0, 1)) * F.pad(valid[:, 1:], (0, 1))
                score_2d = ppl + left + right
                score_2d = score_2d.masked_fill(~allowed_mask_sel.bool(), torch.finfo(score_2d.dtype).min)
                scores_sel = score_2d.reshape(-1)
            elif self.token_method == "ppl":
                # 直接使用已有 token-level CE（已乘 mask）作为分数
                scores_sel = target_token_loss[idx_sub].reshape(-1)
                scores_sel[~allowed_mask_sel.reshape(-1)] = -1e9
            else:
                scores_sel = torch.rand_like(allowed_mask_sel.reshape(-1), dtype=target_token_loss.dtype)
                scores_sel[~allowed_mask_sel.reshape(-1)] = -1e9

            # Select top-n_budget positions globally within selected samples
            topk = torch.topk(scores_sel, k=min(n_budget, scores_sel.numel()), largest=True)
            selection_mask_sel_flat = torch.zeros_like(scores_sel)
            selection_mask_sel_flat[topk.indices] = 1
            selection_mask_sel = selection_mask_sel_flat.reshape(allowed_mask_sel.shape)
            # Scatter back to full-batch shape mask
            selection_mask = torch.zeros_like(allowed_mask_b, dtype=target_token_loss.dtype)
            selection_mask[idx_sub] = selection_mask_sel.to(selection_mask.dtype)

            # Compute selective CE on selected tokens only (global denominator = selected tokens total)
            ce_selected_per_sample = (target_token_loss * selection_mask).sum(dim=1)
            total_selected = selection_mask.sum().clamp_min(1)
            weighted_sum = (ce_selected_per_sample * weights).sum()
            selective_loss = weighted_sum / total_selected

            # Minimal logging for verification
            if getattr(self, "state", None) is not None:
                log_every = int(getattr(self.args, "logging_steps", 10) or 10)
                if log_every <= 0:
                    log_every = 10
                if self.state.global_step % log_every == 0:
                    try:
                        self.log({
                            "budget/total_valid_tokens": float(total_valid_all),
                            "budget/n_budget": float(n_budget),
                        })
                    except Exception:
                        pass

                # Print per-sample kept tokens once when epoch first exceeds 0
                try:
                    epoch_val = getattr(self.state, "epoch", None)
                    if epoch_val is not None and epoch_val > 0 and not self._budget_logged_after_epoch0 and self.is_world_process_zero():
                        kept_per_sample = selection_mask.sum(dim=1).detach().to(dtype=torch.int32).cpu().tolist()
                        valid_per_sample = allowed_mask_b.sum(dim=1).detach().to(dtype=torch.int32).cpu().tolist()
                        # summarize selected data information (sample-level selection by data_ratio)
                        selected_indices = idx_sub.detach().cpu().tolist() if torch.is_tensor(idx_sub) else [int(x) for x in idx_sub]
                        num_selected_samples = len(selected_indices)
                        # print summary for the batch (once)
                        logger.info(
                            f"[BudgetSummary] epoch>{epoch_val:.2f} total_valid={int(total_valid_all)} n_budget={int(n_budget)} "
                            f"num_selected_samples={num_selected_samples} selected_sample_indices={selected_indices}"
                        )
                        logger.info(
                            f"[Budget] epoch>{epoch_val:.2f} kept_tokens_per_sample={kept_per_sample} valid_tokens_per_sample={valid_per_sample}"
                        )
                        self._budget_logged_after_epoch0 = True
                except Exception:
                    pass

            if return_outputs:
                return selective_loss, outputs
            return selective_loss

        # Plugin: wisely (sample selection Q2+Q4 per-batch, then Q2 token pruning on instruction region)
        if getattr(self, "plug", None) == "wisely":
            # First forward already computed as `outputs`. Use it to build ppl/entropy per sample
            logits_full = outputs.logits[:, :-1, :].contiguous()
            ppl_np, ent_np = self._compute_ppl_entropy(
                logits_full, shifted_labels, attention_mask
            )

            # keep_ratio for samples in current batch: (Q2 + Q4) / all
            keep_ratio_samples = float(self.data_ratio)
            ppl_th, ent_th, Q2_mask, Q4_mask = self._dynamic_threshold_search_bisect(ppl_np, ent_np, keep_ratio_samples)

            import numpy as _np
            Q2_idx = _np.where(Q2_mask)[0]
            Q4_idx = _np.where(Q4_mask)[0]
            Qkeep_idx = _np.concatenate([Q2_idx, Q4_idx], axis=0)

            # Note: 暂不打印前置摘要，改为在补足策略执行后统一打印最终摘要，避免 keep=0 的混淆

            # Ensure at least K samples kept, where K = round(data_ratio * batch_size)
            K_keep = max(1, int(round(batch_size * float(self.data_ratio))))
            K_keep = min(K_keep, batch_size)
            extra_added_idx = None  # track augmented indices for Q2/Q4 attribution
            if Qkeep_idx.size < K_keep and ppl_np.size > 0:
                # normalize to [0,1]
                def _norm(x: _np.ndarray):
                    x_min = float(_np.min(x))
                    x_max = float(_np.max(x))
                    if x_max - x_min < 1e-12:
                        return _np.zeros_like(x)
                    return (x - x_min) / (x_max - x_min)

                ppl_n = _norm(ppl_np)
                ent_n = _norm(ent_np)
                s2 = ppl_n - ent_n  # prefer Q2 (high ppl, low entropy)
                s4 = ent_n - ppl_n  # prefer Q4 (high entropy, low ppl)
                wise_score = _np.maximum(s2, s4)

                all_idx = _np.arange(len(ppl_np))
                already = set(Qkeep_idx.tolist())
                remain = _np.array([i for i in all_idx if i not in already], dtype=_np.int64)
                if remain.size > 0:
                    need = K_keep - Qkeep_idx.size
                    # select top-need from remain by wise_score
                    order = _np.argsort(-wise_score[remain])  # descending
                    extra = remain[order[:min(need, remain.size)]]
                    Qkeep_idx = _np.concatenate([Qkeep_idx, extra], axis=0)
                    extra_added_idx = extra

                # 标记发生了补足
                augmented = True
            else:
                augmented = False
                extra_added_idx = _np.array([], dtype=_np.int64)

            # Build subset indices tensor
            idx_sub = torch.tensor(Qkeep_idx if Qkeep_idx.size > 0 else _np.arange(min(1, batch_size)), device=model.device, dtype=torch.long)

            # 统一打印最终摘要（阈值、Q2/Q4原始计数、最终保留数），避免出现 keep=0 误解
            if getattr(self, "state", None) is not None and self.state.global_step % max(1, int(getattr(self.args, "logging_steps", 10) or 10)) == 0:
                try:
                    total_n = len(ppl_np)
                    kept_pre = int(Q2_mask.sum() + Q4_mask.sum())
                    kept_final = int(Qkeep_idx.size)
                    used_kept = int(idx_sub.numel())
                    drop_final = int(max(0, total_n - kept_final))
                    aug_note = f" augmented_to_K={K_keep}" if augmented else ""
                    logger.info(
                        f"[Wisely] step={self.state.global_step} ppl_th={ppl_th:.4f} ent_th={ent_th:.4f} "
                        f"Q2_pre={int(Q2_mask.sum())} Q4_pre={int(Q4_mask.sum())} keep_pre={kept_pre}/{total_n} "
                        f"keep_final={kept_final}/{total_n} used_kept={used_kept} drop_final={drop_final}{aug_note}"
                    )
                    if used_kept != kept_final:
                        logger.warning(
                            f"[Wisely] used_kept({used_kept}) != keep_final({kept_final}), please check selection logic"
                        )
                except Exception:
                    pass

            # Compute token pruning budget for this batch
            total_tokens_batch = attention_mask.sum().item()
            target_keep_tokens = int(round(total_tokens_batch * float(self.data_ratio) * float(self.token_ratio)))
            drop_target = max(0, total_tokens_batch - target_keep_tokens)

            # Tokens dropped by removing Q1+Q3 samples entirely
            keep_mask_batch = torch.zeros_like(attention_mask, dtype=attention_mask.dtype)
            if idx_sub.numel() > 0:
                keep_mask_batch[idx_sub] = attention_mask[idx_sub]
            dropped_tokens_by_samples = int((attention_mask.sum() - keep_mask_batch.sum()).item())

            remaining_drop_for_Q2 = max(0, drop_target - dropped_tokens_by_samples)

            # Compute raw CE per-token on kept set (reuse first forward logits; no second forward)
            idx_sub_dev = idx_sub.to(logits_full.device)
            logits_kept = torch.index_select(logits_full, 0, idx_sub_dev)
            shifted_labels_kept = inputs['labels'][idx_sub][:, 1:].contiguous().detach()
            attn_mask_shift_kept = inputs['attention_mask'][idx_sub][:, 1:].contiguous().detach()
            ce_token_loss_raw = loss_fct(
                logits_kept.reshape(-1, logits_kept.size(-1)),
                shifted_labels_kept.reshape(-1)
            ).reshape(shifted_labels_kept.size())
            base_valid_mask = (attn_mask_shift_kept > 0)

            # If still need to drop tokens, drop from Q2 samples' instruction region by token method ordering
            if remaining_drop_for_Q2 > 0:
                # determine pruning candidates: original Q2 plus augmented Q2-like (s2>=s4) among extras
                kept_indices_np = idx_sub.detach().cpu().numpy()
                map_to_kept = {v: i for i, v in enumerate(kept_indices_np.tolist())}
                # start with original Q2
                q2_all = Q2_idx.copy()
                # add augmented Q2-like if any
                if augmented and extra_added_idx is not None and extra_added_idx.size > 0:
                    # reuse normalization
                    def _norm_local(x: _np.ndarray):
                        x_min = float(_np.min(x))
                        x_max = float(_np.max(x))
                        if x_max - x_min < 1e-12:
                            return _np.zeros_like(x)
                        return (x - x_min) / (x_max - x_min)
                    ppl_n_aug = _norm_local(ppl_np)
                    ent_n_aug = _norm_local(ent_np)
                    s2_aug = ppl_n_aug - ent_n_aug
                    s4_aug = ent_n_aug - ppl_n_aug
                    aug_q2 = extra_added_idx[s2_aug[extra_added_idx] >= s4_aug[extra_added_idx]]
                    if aug_q2.size > 0:
                        q2_all = _np.concatenate([q2_all, aug_q2], axis=0)

                Q2_in_kept = [map_to_kept[i] for i in q2_all if i in map_to_kept]
                if len(Q2_in_kept) > 0:
                    Q2_in_kept_t = torch.tensor(Q2_in_kept, device=model.device, dtype=torch.long)

                    # Build instruction-only mask: here we approximate instruction tokens为 labels中非 -100 且来自 prompt 的位置
                    # 现有模板下，train_on_prompt=False 时，prompt 多为 -100，不剪prompt；若需要仅剪 instruction，可在 Dataset 标注位信息
                    instr_mask = (shifted_labels_kept != -100) & (attn_mask_shift_kept > 0)

                    # Scoring tokens by current token_method
                    if self.token_method == "rho" and hasattr(self, "reference_model") and self.reference_model is not None:
                        ref_outputs_kept = self.reference_model(
                            input_ids=inputs['input_ids'][idx_sub],
                            attention_mask=inputs['attention_mask'][idx_sub],
                            labels=inputs['labels'][idx_sub],
                            return_dict=True
                        )
                        ref_logits_kept = ref_outputs_kept.logits[:, :-1, :]
                        ref_token_loss_kept = loss_fct(
                            ref_logits_kept.reshape(-1, ref_logits_kept.size(-1)),
                            shifted_labels_kept.reshape(-1)
                        ).reshape(shifted_labels_kept.size())
                        ref_token_loss_kept = (ref_token_loss_kept * base_valid_mask).detach().to(dtype=ce_token_loss_raw.dtype)
                        scores_kept = ((ce_token_loss_raw * base_valid_mask) - ref_token_loss_kept)
                    elif self.token_method == "sample_fastv" and getattr(outputs, "attentions", None) is not None:
                        last_attn_kept = outputs.attentions[-1][idx_sub]  # [B', H, Lq, Lk]
                        token_score_per_sample = last_attn_kept.mean(dim=1).mean(dim=1)  # [B', Lk]
                        scores_kept = token_score_per_sample
                    else:
                        scores_kept = (ce_token_loss_raw * base_valid_mask)  # fallback: loss-based

                    # Align scores_kept length to mask length to avoid shape mismatch
                    T_target = base_valid_mask.size(1)
                    if scores_kept.size(1) != T_target:
                        cur_len = scores_kept.size(1)
                        if cur_len < T_target:
                            pad = T_target - cur_len
                            fill_val = torch.finfo(scores_kept.dtype).min
                            scores_kept = F.pad(scores_kept, (0, pad), value=fill_val)
                        else:
                            scores_kept = scores_kept[:, :T_target]

                    # Collect candidate positions only within Q2 samples and instruction mask
                    cand_mask = torch.zeros_like(instr_mask, dtype=torch.bool)
                    cand_mask[Q2_in_kept_t] = instr_mask[Q2_in_kept_t]

                    cand_scores = scores_kept.masked_fill(~cand_mask, torch.finfo(scores_kept.dtype).min)
                    num_cands = int(cand_mask.sum().item())
                    n_drop = min(remaining_drop_for_Q2, num_cands)
                    if n_drop > 0:
                        flat_scores = cand_scores.reshape(-1)
                        topk = torch.topk(flat_scores, k=min(n_drop, flat_scores.numel()), largest=True)
                        drop_flat_idx = topk.indices
                        drop_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
                        drop_mask[drop_flat_idx] = True
                        drop_mask = drop_mask.reshape(cand_scores.shape)
                        # build effective mask without in-place modifying tensors used in graph
                        effective_mask = base_valid_mask.clone()
                        effective_mask[drop_mask] = 0
                    else:
                        effective_mask = base_valid_mask
                else:
                    # no Q2 candidates available while needing to drop tokens
                    try:
                        logger.warning(
                            f"[Wisely] step={getattr(self.state, 'global_step', -1)} no Q2 tokens to prune; skipping token pruning this batch"
                        )
                    except Exception:
                        pass
                    effective_mask = base_valid_mask
            else:
                effective_mask = base_valid_mask

            # Final selective CE over kept set only (apply effective mask)
            ce_token_loss_masked = ce_token_loss_raw * effective_mask
            ce_den_kept = effective_mask.sum().clamp_min(1)
            selective_loss = (ce_token_loss_masked.sum() / ce_den_kept)

            if return_outputs:
                return selective_loss, outputs
            return selective_loss

        # Self-distillation plugin branch
        if self.self_distill:
            # teacher logits f(B)
            teacher_logits = target_logits.detach()  # [B, L-1, V]

            # slice tensors for B'
            input_ids_b = inputs['input_ids'][idx_sub]
            attn_mask_b = inputs['attention_mask'][idx_sub]
            labels_b = inputs['labels'][idx_sub]
            logits_b = outputs.logits[idx_sub, :-1, :]  # f(B')

            # CrossEntropy on B' (average over valid positions)
            shifted_labels_b = labels_b[:, 1:].contiguous().detach()
            attn_mask_shift_b = attn_mask_b[:, 1:].contiguous().detach()
            ce_token_loss_b = loss_fct(
                logits_b.reshape(-1, logits_b.size(-1)),
                shifted_labels_b.reshape(-1)
            ).reshape(shifted_labels_b.size())
            ce_token_loss_b = ce_token_loss_b * attn_mask_shift_b
            ce_den = attn_mask_shift_b.sum().clamp_min(1)
            ce_loss = ce_token_loss_b.sum() / ce_den

            # Build selection mask WITHIN B' using current token_method (exact proportion top-k)
            sel_token_loss_b = ce_token_loss_b
            # Protect special tokens (e.g., EOS, PAD, all_special_ids)
            allowed_mask_b = (attn_mask_shift_b > 0)
            try:
                special_ids = getattr(self.processing_class, "all_special_ids", None)
            except Exception:
                special_ids = None
            if special_ids is not None and isinstance(special_ids, (list, tuple)) and len(special_ids) > 0:
                special_mask = torch.zeros_like(allowed_mask_b, dtype=torch.bool)
                cur_tokens = input_ids_b[:, 1:]
                for sid in special_ids:
                    special_mask |= (cur_tokens == sid)
                allowed_mask_b = allowed_mask_b & (~special_mask)
            valid_mask_flat = allowed_mask_b.reshape(-1)
            num_valid = int(valid_mask_flat.sum().item())
            k_sel = max(1, int(round(num_valid * float(self.token_ratio))))

            if self.token_method == "random":
                selection_mask_b_flat = torch.zeros_like(valid_mask_flat, dtype=sel_token_loss_b.dtype)
                if num_valid > 0:
                    valid_idx_flat = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                    choose = valid_idx_flat[torch.randperm(num_valid, device=valid_idx_flat.device)[:k_sel]]
                    selection_mask_b_flat[choose] = 1
                selection_mask_b = selection_mask_b_flat.reshape(sel_token_loss_b.shape)
            elif self.token_method == "rho" and hasattr(self, "reference_model") and self.reference_model is not None:
                ref_outputs_b = self.reference_model(
                    input_ids=input_ids_b,
                    attention_mask=attn_mask_b,
                    labels=labels_b,
                    return_dict=True
                )
                ref_logits_b = ref_outputs_b.logits[:, :-1, :]
                ref_token_loss_b = loss_fct(
                    ref_logits_b.reshape(-1, ref_logits_b.size(-1)),
                    shifted_labels_b.reshape(-1)
                ).reshape(shifted_labels_b.size())
                ref_token_loss_b = (ref_token_loss_b * attn_mask_shift_b).detach().to(dtype=sel_token_loss_b.dtype)
                excess_b = (sel_token_loss_b - ref_token_loss_b).reshape(-1)
                selection_mask_b_flat = torch.zeros_like(excess_b, dtype=sel_token_loss_b.dtype)
                if num_valid > 0:
                    valid_idx_flat = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                    valid_excess = excess_b[valid_idx_flat]
                    topk_idx = torch.topk(valid_excess, k=min(k_sel, valid_excess.numel()), largest=True).indices
                    choose = valid_idx_flat[topk_idx]
                    selection_mask_b_flat[choose] = 1
                selection_mask_b = selection_mask_b_flat.reshape(sel_token_loss_b.shape)
            else:
                selection_mask_b = (attn_mask_shift_b > 0).to(sel_token_loss_b.dtype)

            # KL1: KL(f(B'), f(B).detach) over selected positions in B'
            T = float(self.distill_temperature)
            kl1 = mmd_kl_divergence(logits_b, teacher_logits, kernel=None)
            print("MMD KL divergence:", kl1.item())
            
            # Build B'' inputs by masking selected positions (drop tokens): set mask 0, input_ids to pad
            pad_id = getattr(self.processing_class, "pad_token_id", 0)
            input_ids_b2 = input_ids_b.clone()
            attn_mask_b2 = attn_mask_b.clone()
            # map selection_mask (L-1) to input positions [1:]
            mask_input_dim = torch.zeros_like(attn_mask_b)
            mask_input_dim[:, 1:] = selection_mask_b.to(mask_input_dim.dtype)
            pruned_positions = (mask_input_dim > 0) & (attn_mask_b2 > 0)
            input_ids_b2[pruned_positions] = pad_id
            attn_mask_b2[pruned_positions] = 0

            outputs_b2 = model(
                input_ids=input_ids_b2,
                attention_mask=attn_mask_b2,
                labels=labels_b,
                return_dict=True
            )
            logits_b2 = outputs_b2.logits[:, :-1, :]

            # KL2 over selected positions only
            sel_positions = selection_mask_b > 0
            if sel_positions.any():
                log_p2 = F.log_softmax(logits_b2[sel_positions] / T, dim=-1)
                q2 = F.softmax(logits_b[sel_positions] / T, dim=-1)
                kl2 = F.kl_div(log_p2, q2, reduction='batchmean')
            else:
                kl2 = torch.zeros((), device=model.device, dtype=ce_loss.dtype)

            # Scale KL terms by T^2 as in Hinton et al. KD to match gradient magnitudes
            T_squared = T * T
            kl1 = kl1 * T_squared
            kl2 = kl2 * T_squared

            # CE on selected tokens only
            ce_selected = (ce_token_loss_b * selection_mask_b).sum()
            ce_den_sel = selection_mask_b.sum().clamp_min(1)
            ce_loss_sel = ce_selected / ce_den_sel
            total_loss = ce_loss_sel + self.distill_lambda_1 * kl1 + self.distill_lambda_2 * kl2

            # Logging for verification
            try:
                selected_tokens_full = float('nan')
            except Exception:
                selected_tokens_full = float('nan')
            try:
                selected_tokens_subset = selection_mask_b.sum().detach().to(dtype=torch.float32).item()
            except Exception:
                selected_tokens_subset = float('nan')
            try:
                valid_tokens_full = float('nan')
            except Exception:
                valid_tokens_full = float('nan')
            try:
                valid_tokens_subset = attn_mask_shift_b.sum().detach().to(dtype=torch.float32).item()
            except Exception:
                valid_tokens_subset = float('nan')

            if getattr(self, "state", None) is not None:
                log_every = int(getattr(self.args, "logging_steps", 10) or 10)
                if log_every <= 0:
                    log_every = 10
                if self.state.global_step % log_every == 0:
                    self.log({
                        "distill/ce": float(ce_loss_sel.detach().cpu().item()),
                        "distill/kl1": float(kl1.detach().cpu().item()),
                        "distill/kl2": float(kl2.detach().cpu().item()),
                        "distill/T": float(T),
                        "distill/data_ratio": float(keep_ratio),
                        "distill/token_ratio": float(self.token_ratio),
                        "distill/batch_size": float(batch_size),
                        "distill/bprime_size": float(subset_size),
                        "distill/selected_tokens_full": float(selected_tokens_full),
                        "distill/valid_tokens_full": float(valid_tokens_full),
                        "distill/selected_tokens_subset": float(selected_tokens_subset),
                        "distill/valid_tokens_subset": float(valid_tokens_subset),
                    })
                    logger.info(
                        f"[SelfDistill] step={self.state.global_step} T={T} data_ratio={keep_ratio} token_ratio={self.token_ratio} "
                        f"CE_sel={ce_loss_sel.detach().cpu().item():.6f} KL1={kl1.detach().cpu().item():.6f} KL2={kl2.detach().cpu().item():.6f} "
                        f"B={batch_size} B'={subset_size} sel_tokens_subset={selected_tokens_subset}/{valid_tokens_subset}"
                    )

            if return_outputs:
                return total_loss, outputs
            return total_loss

        # Non-distill branch: select tokens within B' only and compute CE on selected tokens
        input_ids_b = inputs['input_ids'][idx_sub]
        attn_mask_b = inputs['attention_mask'][idx_sub]
        labels_b = inputs['labels'][idx_sub]
        shifted_labels_b = labels_b[:, 1:].contiguous().detach()
        attn_mask_shift_b = attn_mask_b[:, 1:].contiguous().detach()
        weights_b = weights[idx_sub]
        # Protect special tokens for non-distill path
        allowed_mask_b = (attn_mask_shift_b > 0)
        try:
            special_ids = getattr(self.processing_class, "all_special_ids", None)
        except Exception:
            special_ids = None
        if special_ids is not None and isinstance(special_ids, (list, tuple)) and len(special_ids) > 0:
            special_mask = torch.zeros_like(allowed_mask_b, dtype=torch.bool)
            cur_tokens = input_ids_b[:, 1:]
            for sid in special_ids:
                special_mask |= (cur_tokens == sid)
            allowed_mask_b = allowed_mask_b & (~special_mask)

        if self.token_method == "random": 
            flat_target_loss = target_token_loss[idx_sub].reshape(-1)
            flat_attention_mask = allowed_mask_b.reshape(-1)
            valid_indices = torch.nonzero(flat_attention_mask > 0, as_tuple=True)[0]
            selection_mask_flat = torch.zeros_like(flat_target_loss)
            if len(valid_indices) > 0:
                k_sel = max(1, int(round(len(valid_indices) * self.token_ratio)))
                chosen = valid_indices[torch.randperm(len(valid_indices), device=valid_indices.device)[:k_sel]]
                selection_mask_flat[chosen] = 1
            selection_mask = selection_mask_flat.reshape(attn_mask_shift_b.shape)
            selective_token_loss = target_token_loss[idx_sub] * selection_mask
            
        elif self.token_method == "rho": 
            self.reference_model = self.reference_model.to(model.device)
            ref_outputs = self.reference_model(
                input_ids=input_ids_b,
                attention_mask=attn_mask_b,
                labels=labels_b,
                return_dict=True
            )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_token_loss = loss_fct(
                ref_logits.reshape(-1, ref_logits.size(-1)),
                shifted_labels_b.reshape(-1)
            ).reshape(shifted_labels_b.size())
            
            ref_token_loss = (ref_token_loss * attn_mask_shift_b).detach().to(dtype=target_token_loss.dtype)

            # Calculate excess loss (target - reference)
            excess_loss = target_token_loss[idx_sub] - ref_token_loss
            flat_excess_loss = excess_loss.reshape(-1)
            flat_attention_mask = allowed_mask_b.reshape(-1)
            valid_indices = torch.nonzero(flat_attention_mask > 0, as_tuple=True)[0]
            selection_mask_flat = torch.zeros_like(flat_excess_loss)
            if len(valid_indices) > 0:
                k_sel = max(1, int(round(len(valid_indices) * self.token_ratio)))
                valid_excess = flat_excess_loss[valid_indices]
                topk_idx = torch.topk(valid_excess, k=min(k_sel, valid_excess.numel()), largest=True).indices
                chosen = valid_indices[topk_idx]
                selection_mask_flat[chosen] = 1
            selection_mask = selection_mask_flat.reshape(attn_mask_shift_b.shape)
            selective_token_loss = target_token_loss[idx_sub] * selection_mask
        elif self.token_method == "wise":
            valid = allowed_mask_b.to(dtype=target_token_loss.dtype)
            per_token_loss = target_token_loss[idx_sub]
            ppl = torch.exp(per_token_loss) * valid
            left = F.pad(ppl[:, :-1], (1, 0)) * F.pad(valid[:, :-1], (1, 0))
            right = F.pad(ppl[:, 1:], (0, 1)) * F.pad(valid[:, 1:], (0, 1))
            score_2d = ppl + left + right
            score_2d = score_2d.masked_fill(~allowed_mask_b.bool(), torch.finfo(score_2d.dtype).min)
            flat_scores = score_2d.reshape(-1)
            valid_indices = torch.nonzero(allowed_mask_b.reshape(-1) > 0, as_tuple=True)[0]
            selection_mask_flat = torch.zeros_like(flat_scores)
            if len(valid_indices) > 0:
                k_sel = max(1, int(round(len(valid_indices) * self.token_ratio)))
                valid_scores = flat_scores[valid_indices]
                topk_idx = torch.topk(valid_scores, k=min(k_sel, valid_scores.numel()), largest=True).indices
                chosen = valid_indices[topk_idx]
                selection_mask_flat[chosen] = 1
            selection_mask = selection_mask_flat.reshape(attn_mask_shift_b.shape)
            selective_token_loss = target_token_loss[idx_sub] * selection_mask
        elif self.token_method == "sample_fastv":
            # print(self.token_method)
            last_attn = outputs.attentions[-1]   # [B, H, Lq, Lk]
            last_attn_sub = last_attn[idx_sub]
            # print("last_attn_sub shapel:", last_attn_sub.shape) # 
            if last_attn is None:
                raise RuntimeError("outputs.attentions[-1] is None")

            device = last_attn_sub.device
            B, H, Lq, Lk = last_attn_sub.shape

            token_score_per_sample = last_attn_sub.mean(dim=1).mean(dim=1)  # [B, Lk]
            assert token_score_per_sample.shape == (B, Lk)

            # L_mask = attn_mask_shift_b.shape[1]
            # if L_mask != Lk:
            #     new_len = min(L_mask, Lk)
            #     # print(f"Dimension mismatch: attention Lk={Lk}, mask L_mask={L_mask}. Aligning to {new_len}.")
            #     token_score_per_sample = token_score_per_sample[:, :new_len]  # [B, new_len]
            #     attn_mask = attn_mask_shift_b[:, :new_len].to(device)
            #     Lk = new_len
            # else:
            #     attn_mask = attn_mask_shift_b.to(device)  # [B, Lk]
            Lk = Lk - 1
            token_score_per_sample = token_score_per_sample[:,1:] # drop start-of-sentence
            attn_mask = allowed_mask_b.to(device)

            selection_mask = torch.zeros((B, Lk), dtype=token_score_per_sample.dtype, device=device)
            for i in range(B):
                scores = token_score_per_sample[i]    # [Lk]
                mask_i = attn_mask[i].to(device)      # [Lk], 0/1
                valid_idx = torch.nonzero(mask_i > 0, as_tuple=True)[0]

                if valid_idx.numel() == 0:
                    print(f"Warning: sample {i} has no valid indices in mask; keeping index 0")
                    selection_mask[i, 0] = 1.0
                    continue
                
                k_sel = max(1, int(round(valid_idx.numel() * self.token_ratio)))
                k_sel = min(k_sel, valid_idx.numel())

                valid_scores = scores[valid_idx]
                try:
                    topk_local = torch.topk(valid_scores, k=k_sel, largest=True).indices  # indices in valid_scores
                    topk_local = topk_local.to(device)
                except RuntimeError as e:
                    print(f"GPU topk failed for sample {i} (falling back to CPU). Error: {e}")
                    topk_local = torch.topk(valid_scores.detach().cpu(), k=k_sel, largest=True).indices.to(device)

                chosen_idx = valid_idx[topk_local]  
                chosen_idx = torch.clamp(chosen_idx, 0, Lk - 1)
                selection_mask[i, chosen_idx] = 1.0
                
            selective_token_loss = target_token_loss[idx_sub] * selection_mask

        elif self.token_method == "sample_sparsevlm":
            last_attn = outputs.attentions[-1]   # [B, H, Lq, Lk]
            last_attn_sub = last_attn[idx_sub]
            # print("last_attn_sub shape", last_attn_sub.shape)

            if last_attn_sub is None:
                raise RuntimeError("outputs.attentions[-1] is None")

            device = last_attn_sub.device
            B, H, Lq, Lk = last_attn_sub.shape

            p_score = last_attn_sub.mean(dim=1).mean(dim=1)  # [B, Lk]

            last_hidden = outputs.hidden_states[-1]  # [B, L, D]
            last_hidden_sub = last_hidden[idx_sub]   # [B, Lk, D]
            normed = F.normalize(last_hidden_sub, dim=-1)
            S = torch.matmul(normed, normed.transpose(-1, -2))  # [B, Lk, Lk]
            s_score = S.mean(dim=1)  # [B, Lk]

            alpha = 0.5
            token_score_per_sample = alpha * p_score + (-1 + alpha) * s_score  # [B, Lk]

            Lk = Lk - 1
            token_score_per_sample = token_score_per_sample[:,1:] # drop start-of-sentence
            attn_mask = allowed_mask_b.to(device)
            # print("token_score_per_sample shape",token_score_per_sample.shape)
            
            P_mean = last_attn_sub.mean(dim=(0, 1))  # [Lq, Lk]
            try:
                rank = torch.linalg.matrix_rank(P_mean).item()
            except RuntimeError:
                # fallback: approximate rank via svd_lowrank
                U, Svals, V = torch.svd_lowrank(P_mean, q=min(32, min(Lq, Lk)))
                rank = (Svals > 1e-5).sum().item()
            
            # paper这 N 是max number of pruning tokens
            N = int(Lk - rank)  
            N = max(1, N)       

            selection_mask = torch.zeros((B, Lk), dtype=token_score_per_sample.dtype, device=device)

            for i in range(B):
                scores = token_score_per_sample[i]    # [Lk]
                mask_i = attn_mask[i].to(device)      # [Lk], 0/1
                valid_idx = torch.nonzero(mask_i > 0, as_tuple=True)[0]

                if valid_idx.numel() == 0:
                    print(f"Warning: sample {i} has no valid indices in mask; keeping index 0")
                    selection_mask[i, 0] = 1.0
                    continue
                
                sorted_idx = torch.argsort(scores, descending=False)  
                n_valid = (mask_i > 0).sum().item()
                n_del = min(N, max(0, n_valid - 1))  
                prune_idx = sorted_idx[:n_del]
                # keep_idx = sorted_idx[:n_del]
                # keep_idx = torch.unique(keep_idx) 

                selection_mask[i, prune_idx] = 1.0

            selective_token_loss = target_token_loss[idx_sub] * selection_mask    
            # print("target_token_loss shape",target_token_loss.shape)
            # print("selection_mask shape", selection_mask.shape)        
            # last_attn = outputs.attentions[-1]   # [B, H, Lq, Lk]
            # last_attn_sub = last_attn[idx_sub]
            # B, H, L, _ = last_attn_sub.shape
            # print("last_attn_sub shape: ", last_attn_sub.shape)
            # last_hidden = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
            # last_hidden_sub = last_hidden[idx_sub]
            # print("last_hidden_sub shape: ", last_hidden_sub.shape)
            
            # p_score = last_attn_sub.mean(dim=1).mean(dim=1) # [B, L]
            # normed = F.normalize(last_hidden_sub, dim=-1)
            # S = torch.matmul(normed, normed.transpose(-1,-2)) # [B, L, L]
            # s_score = S.mean(dim=1) # [B, L]
                
            # # can use α*p_score + (1-α)*s_score, here set α = 0.5    
            # M = 0.5 * (p_score + s_score) # [B,L]
            
            # L_mask = attn_mask_shift_b.shape[1]
            # if L_mask != Lk:
            #     new_len = min(L_mask, Lk)
            #     token_score_per_sample = token_score_per_sample[:, :new_len]  # [B, new_len]
            #     attn_mask = attn_mask_shift_b[:, :new_len].to(device)
            #     Lk = new_len
            # else:
            #     attn_mask = attn_mask_shift_b.to(device)  # [B, Lk]
                
                
            # P_collapse = last_attn_sub.reshape(-1,L)
            # P_mean = last_attn.mean(dim=(0,1)) # [L,L]
            
            # # maybe very slow, can be replaced by torch.svd_lowrank
            # rank = torch.linalg.matrix_rank(P_mean) 
            # N = int(L-rank) # max number of pruning tokens
            # N = max(1, N) 
            
            # selection_mask = torch.zeros((B, L), dtype=M.dtype, device=M.device)
            # for i in range(B):
            #     scores = M[i]  # [L]
            #     sorted_idx = torch.argsort(scores, descending=False)
            #     prune_idx = sorted_idx[:N]
            #     keep_idx = sorted_idx[N:]
            #     selection_mask[i, keep_idx] = 1.0

            # # ---- Step 6: 结合 loss 计算最终 loss ----
            # token_loss = target_token_loss[idx_sub][:, :L]  # [B, L]
            # selective_token_loss = token_loss * selection_mask
            
            
        
        
        # Normalize by the number of selected tokens within B'
        num_selected = selection_mask.sum().to(dtype=target_token_loss.dtype)
        selective_token_loss = selective_token_loss.sum(dim=1).to(dtype=target_token_loss.dtype) * weights_b
        if num_selected > 0:
            selective_loss = selective_token_loss.sum() / num_selected
        else:
            selective_loss = (target_token_loss[idx_sub] * attn_mask_shift_b).sum() / (attn_mask_shift_b.sum() + 1e-10)
        
        if return_outputs:
            return selective_loss, outputs
        return selective_loss
        # return torch.zeros_like(selective_loss, requires_grad=True)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")