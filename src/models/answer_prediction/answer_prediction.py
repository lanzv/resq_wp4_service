import logging
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, StoppingCriteria, StoppingCriteriaList
from src.form.form_definition import FormDefinition
from src.form.form_questions import FormQuestions
from src.form.question_confidences import QuestionConfidences
from .utils import load_model, generate_prompt, prompt_model, parse_output, generate_answer, ParsingError
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from functools import partial
from textwrap import dedent
import torch
import peft.tuners.lora.layer as lora_layer
from tqdm import tqdm
import transformers
import json
import re
import numpy as np

class AnswerPredictionModel:
    def __init__(self, model_path = "../models/BioMistral-7B"):
        self.model, self.tokenizer = load_model(model_path=model_path)
        self.form_questions = FormQuestions()
        self.form_definition = FormDefinition()
        self.question_confidences = QuestionConfidences()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        peft_config = LoraConfig(
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)


    def predict(self, request):
        raw_test, ids, question_ids = self.prepare_dataset(request, return_ids=True)

        predictions = {}
        prediction_question_ids = {}
        model_confidences = {}
        for qaid, question_id, messages in zip(ids, question_ids, raw_test["messages"]):
            if question_id in self.form_definition.possible_options:
                model_answer, score = get_answer(
                    messages=messages,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    num_beams=1
                )
                predictions[qaid] = parse_output(
                            model_answer,
                            self.form_definition,
                            question_id,
                            messages
                    )
                prediction_question_ids[qaid] = question_id
                model_confidences[qaid] = score
            else:
                predictions[qaid] = None
                prediction_question_ids[qaid] = question_id
                model_confidences[qaid] = 0.0
        response = {"predictions": []}
        for qaid in ids:
            q_conf = self.question_confidences.get_answer_prediction_confidence(
                prediction_question_ids[qaid], request["language"]
                )
            response["predictions"].append(
                {
                    "question_id": prediction_question_ids[qaid],
                    "question_confidence": q_conf["question_confidence"],
                    "question_frequency": q_conf["question_frequency"],
                    "enumeration_value_id": predictions[qaid] if not isinstance(predictions[qaid], ParsingError) else None,
                    "prediction_confidence": model_confidences[qaid]
                }
            )
        return response

    def prepare_dataset(self, request, return_ids=False): 
        # Prepare the dataset
        preprocessed_dataset = []
        ids = []
        question_ids = []
        for i, qa in enumerate(request["questions"]):
            prompt = generate_prompt(
                self.form_definition, 
                "",  # No few-shot examples for fine-tuning
                self.form_questions.load_question(qa["question_id"]),
                qa["evidences"], 
                qa["question_id"]
            )
            messages = {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                        {
                            "role": "assistant",
                            "content": ' '
                        },
                    ]
                }

            preprocessed_dataset.append(messages)
            ids.append(qa["question_id"]+"#"+str(i))
            question_ids.append(qa["question_id"])
        if return_ids:
            return Dataset.from_list(preprocessed_dataset), ids, question_ids
        return Dataset.from_list(preprocessed_dataset)


def apply_chat_template(messages, include_answer = True, inst_start="[INST]", inst_end="[\INST]"):
    """
    Formats a list of role-based messages into a chat-style template.

    Parameters:
    ----------
    messages : list of dict
        List of dictionaries containing 'content' and 'role' keys.

    Returns:
    -------
    formatted_message : str
        A single formatted message with roles and content combined.
    """
    formatted_parts = []
    last_assistant = 0
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted_parts.append(f"{inst_start} {content} {inst_end}")
        elif role == "assistant":
            last_assistant = i
            formatted_parts.append(f"{content}")
        else:
            formatted_parts.append(f"{content}")
    if not include_answer:
        return " ".join(formatted_parts[:last_assistant])
    return " ".join(formatted_parts)


def get_answer(messages, model, tokenizer, num_beams=None):
    prompt = apply_chat_template(messages, include_answer=False)    
    tokenized_prompt = tokenizer([prompt], add_special_tokens=True, return_tensors="pt")
    output = model.generate(
        input_ids = tokenized_prompt.input_ids.to(model.device),
        attention_mask = tokenized_prompt.attention_mask.to(model.device),
        output_scores=False, output_logits=True, return_dict_in_generate=True,
        do_sample=False,
        #num_beams=num_beams,
        num_return_sequences=1,
        max_new_tokens=50,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    model_answer = tokenizer.decode(
            output.sequences[0], skip_special_tokens=True
        )[len(prompt):].strip()

    probs = torch.nn.functional.softmax(torch.stack(output.logits), dim=-1)
    max_probs, _ = probs.max(dim=-1)
    scores = max_probs.mean(dim=0).tolist()
    return model_answer, scores[0]