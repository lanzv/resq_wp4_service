import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, DistilBertForQuestionAnswering
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from src.form.question_confidences import QuestionConfidences
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
from src.form.form_questions import FormQuestions
from .models_for_evidence_extraction import DistilBertForEvidenceExtraction, BertForEvidenceExtraction, XLMRobertaForEvidenceExtraction



class EvidenceExtractionModel:
    def __init__(self, Evidence_Extraction_Model=DistilBertForEvidenceExtraction, model_name = "../models/distilbert-base-multilingual-cased", max_length=256):
        self.form_questions = FormQuestions()
        self.question_confidences = QuestionConfidences()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Evidence_Extraction_Model.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.hide_token_id = self.tokenizer.convert_tokens_to_ids("[HIDE]")

    
    def predict(self, request, batch_size=16, disable_tqdm=True):
        # Preprocess test data and create dataset
        test_examples, ids = self.preprocess_data(request)
        test_dataset = EvidenceDataset(test_examples, self.tokenizer, self.hide_token_id, max_length=self.max_length, inference=True)
        self.model.eval()
        predictions = {}
        prediction_question_ids = {}

        # Predict in batches
        dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=inference_collate_fn)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction", disable=disable_tqdm):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sequence_ids = batch["sequence_ids"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                start_probs, end_probs = outputs["start_probs"].cpu(), outputs["end_probs"].cpu()
                seq_length = start_probs.size(1)

                for i, example_id in enumerate(batch["id"]):
                    prediction_question_ids[example_id] = batch["question_id"][i]

                    if not example_id in predictions:
                        predictions[example_id] = []
                    cur_input_ids = input_ids[i].clone()  # Clone for hiding spans during iteration
                    cur_attention_mask = attention_mask[i]
                    hard_limit_counter = 0
                    while True:
                        # Get span prediction
                        start_pred, end_pred, score = get_span_prediction(
                            start_probs[i], end_probs[i], seq_length, sequence_ids[i]
                        )

                        # Break if no valid span is found
                        if start_pred == 0 and end_pred == 0:
                            break
                        hard_limit_counter += 1
                        if hard_limit_counter % 200 == 0:
                            break

                        # Extract text using offset mapping
                        offset_mapping = batch["offset_mapping"][i]
                        span_text = batch["context"][i][
                            offset_mapping[start_pred][0]:offset_mapping[end_pred][1]
                        ]
                        predictions[example_id].append({"answer_start": offset_mapping[start_pred][0], "text": span_text, "answer_type": "single", "model_confidence": score})

                        # Hide the current span in input IDs and re-run model
                        cur_input_ids[start_pred:end_pred + 1] = self.hide_token_id
                        sub_outputs = self.model(input_ids=cur_input_ids.unsqueeze(0), attention_mask=cur_attention_mask.unsqueeze(0))
                        start_probs[i], end_probs[i] = sub_outputs["start_probs"].cpu(), sub_outputs["end_probs"].cpu()

        response = {"predictions": []}
        for qaid in ids:
            q_conf = self.question_confidences.get_evidence_extraction_confidence(
                prediction_question_ids[qaid], request["language"]
                )
            response["predictions"].append(
                {
                    "question_id": prediction_question_ids[qaid],
                    "question_confidence": q_conf["question_confidence"],
                    "question_frequency": q_conf["question_frequency"],
                    "answers": [],
                }
            )
            for ans in predictions[qaid]:
                response["predictions"][-1]["answers"].append(
                    {
                        "answer_start": ans["answer_start"].item(),
                        "text": ans["text"],
                        "answer_type": ans["answer_type"],
                        "prediction_confidence": ans["model_confidence"].item()
                    }
                )

        return response



    def preprocess_data(self, request):
        preprocessed_samples = []
        context = request["context"]
        ids = []
        for i, question_id in enumerate(request["question_ids"]):
            preprocessed_samples.append(
                {
                    "question": self.form_questions.load_question(question_id),
                    "context": context,
                    "id": str(i),
                    "question_id": question_id
                }
            )
            ids.append(str(i))
        return preprocessed_samples, ids
        

  
class EvidenceDataset(Dataset):
    def __init__(self, examples, tokenizer, hide_token_id, max_length, inference=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.hide_token_id = hide_token_id
        for example in examples:
            encoded = self.tokenizer(
                example["question"],
                example["context"],
                truncation="only_second",
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                stride=20,
                return_tensors="pt",
                padding="max_length",
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            offset_mapping = encoded["offset_mapping"]

            for i in range(len(input_ids)):
                chunk_input_ids = input_ids[i].clone()
                chunk_offset_mapping = offset_mapping[i].cpu().numpy()
                
                self.data.append({
                    "input_ids": chunk_input_ids.squeeze(0),
                    "attention_mask": attention_mask[i].squeeze(0),
                    "offset_mapping": offset_mapping[i].squeeze(0),
                    "id": example["id"],
                    "context": example["context"],
                    "question_id": example["question_id"],
                    "sequence_ids": [0 if v is None else v for v in encoded.sequence_ids(i)]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_token_position_from_char_offset(self, offset_mapping, start_offsets, end_offsets, sequence_ids, chunk_input_ids):
        start_token, end_token = None, None
        start_label = None
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if sequence_ids[token_idx] != 1 or chunk_input_ids[token_idx] == self.hide_token_id:
                continue
            if start_token is None:
                start_token = token_idx
            end_token = token_idx
            for start_offset in start_offsets:
                if start_char <= start_offset < end_char:
                    start_label = token_idx
            for end_offset in end_offsets:
                if start_char < end_offset <= end_char:
                    if start_label is None:
                        return start_token, token_idx
                    else:
                        return start_label, token_idx
        if not start_label is None:
            return start_label, end_token
        return 0, 0 # CLS token
    

def inference_collate_fn(batch):
    # Filter out any `None` entries if they exist
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("All batch entries are None or invalid.")


    # Use default_collate to batch tensor data
    collated_batch = {
        "input_ids": default_collate([item["input_ids"] for item in batch]),
        "attention_mask": default_collate([item["attention_mask"] for item in batch]),
        "offset_mapping": [item["offset_mapping"] for item in batch], 
        "id": [item["id"] for item in batch],
        "question_id": [item["question_id"] for item in batch],
        "context": [item["context"] for item in batch],
        "sequence_ids": [item["sequence_ids"] for item in batch]
    }

    return collated_batch




def get_span_prediction(start_probs, end_probs, seq_length, sequence_ids=None):
    # Compute the outer product
    product_matrix = torch.outer(start_probs, end_probs)  # Shape: (seq_length, seq_length)

    # Create masks
    triu_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool))  # n <= m
    zero_mask = torch.ones(seq_length, seq_length, dtype=torch.bool)
    zero_mask[0, 1:] = 0  # If n == 0, then m must be 0
    zero_mask[1:, 0] = 0  # If m == 0, then n must be 0

    # Combine masks
    valid_mask = triu_mask & zero_mask

    # Add sequence_ids condition if provided
    if sequence_ids is not None:
        sequence_ids = torch.tensor(sequence_ids, dtype=torch.bool)  # Convert to tensor if necessary
        seq_mask = torch.outer(sequence_ids, sequence_ids)  # Mask valid spans based on sequence_ids
        valid_mask &= seq_mask  # Update the valid mask to consider only valid sequence_ids
        valid_mask[0, 0] = True

    # Apply the mask to the product matrix
    product_matrix = product_matrix * valid_mask

    # Find the maximum value and its indices
    max_val, max_idx = product_matrix.view(-1).max(dim=0)  # Flatten and find max
    start_pred, end_pred = divmod(max_idx.item(), seq_length)  # Convert back to 2D indices

    return start_pred, end_pred, max_val
