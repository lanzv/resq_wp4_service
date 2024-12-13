import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DistilBertForQuestionAnswering, BertForQuestionAnswering, XLMRobertaForQuestionAnswering
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering
from torch.utils.data._utils.collate import default_collate


class DistilBertForEvidenceExtraction(DistilBertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        # Use the parent class's forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        # Extract logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Compute probabilities using softmax
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        # Return enhanced output with probabilities
        return {
            "loss": outputs.loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_probs": start_probs,
            "end_probs": end_probs,
        }





class BertForEvidenceExtraction(BertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        # Use the parent class's forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        # Extract logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Compute probabilities using softmax
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        # Return enhanced output with probabilities
        return {
            "loss": outputs.loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_probs": start_probs,
            "end_probs": end_probs,
        }



class XLMRobertaForEvidenceExtraction(XLMRobertaForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        # Use the parent class's forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        # Extract logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Compute probabilities using softmax
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        # Return enhanced output with probabilities
        return {
            "loss": outputs.loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_probs": start_probs,
            "end_probs": end_probs,
        }