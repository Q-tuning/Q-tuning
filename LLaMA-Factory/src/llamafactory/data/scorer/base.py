import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import torch
import copy

logger = logging.getLogger(__name__)

class Scorer(object):
    def __init__(self, model, tokenizer, **kwargs):
        # Automatically detecte device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logger.info(f"Using device: {self.device}")
        self.model = model
        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.padding_side = "left"

    def infer_score(self, user_input: str):
        max_length = 2
        
        # Encode the input as a tensor and move it to the device
        inputs = self.tokenizer(user_input, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        score_npys = []
        for i in range(input_ids.shape[0]):
            try:
                # Move logits to CPU and convert them to a NumPy array
                logprobs_list = outputs.scores[0][i].detach().cpu().numpy()
            except IndexError:
                return 3.0

            score_logits = []
            indices = np.where(logprobs_list != -np.inf)[0]
            id2score = indices[:6] if len(indices) >= 6 else indices
            score_template = np.arange(1, len(id2score) + 1)
            # score_template = np.array([1, 2, 3, 4, 5, 6])
            for k in id2score:
                try:
                    score_logits.append(logprobs_list[k])
                except KeyError:
                    return 3.0

            score_logits = np.array(score_logits)
            score_npy = softmax(score_logits, axis=0)
            score_npy = score_npy * score_template

            score_npy = np.sum(score_npy, axis=0)
            score_npys.append(score_npy)

        return score_npys

    def infer_complexity(self, input_text: str):
        complexity_template = self.complexity_template
        user_input = []
        for i in range(len(input_text)):
            user = complexity_template.format(instruction=input_text[i])
            user_input.append(user)

        return self.infer_score(user_input)

    def infer_quality(self, input_text: str, resp_text: str):
        quality_template = self.quality_template
        user_input = []
        for i in range(len(input_text)):
            user = quality_template.format(instruction=input_text[i], output=resp_text[i])
            user_input.append(user)

        return self.infer_score(user_input)

    @property
    def id2score(self):
        raise NotImplementedError

    @property
    def complexity_template(self):
        raise NotImplementedError

    @property
    def quality_template(self):
        raise NotImplementedError
