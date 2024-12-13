from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from src.form.form_definition import FormDefinition
import string
import random


class ParsingError:
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message
    
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda")
    return model, tokenizer



def prompt_model(model, tokenizer, prompt, question_id, few_shot=True):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    if not few_shot:
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.8,
            top_k=50,
            penalty_alpha=0.6,
            num_beams=5,
            max_new_tokens=50,
        )
    else:
        output_ids = model.generate(**inputs,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.97,
            top_k=0,
            penalty_alpha=0.6,
            num_beams=1,
            max_new_tokens=50,
        )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs




def parse_output(outputs, form_definition, question_id, prompt):
    if not question_id in form_definition.possible_options:
        return None
    outputs = outputs.strip()
    if len(outputs) == 0 and not form_definition.question_includes_datatype(question_id, "string"):
        err_msg = "Empty answer of question {}".format(question_id)
        logging.warning(err_msg)
        return ParsingError(err_msg)
    if form_definition.question_includes_datatype(question_id, "null") and (outputs == "I don’t know"):
        return None
    elif form_definition.question_includes_datatype(question_id, "enum"):
        option_dict = {}
        for letter, option in zip(string.ascii_uppercase, [str(op) for op in form_definition.possible_options[question_id] if op != None]):
            option_dict[letter] = str(option)
        if outputs in option_dict:
            return option_dict[outputs]
        else:
            err_msg = "Option '{}' is not one of the possible items of options {} in prompt {}".format(outputs, option_dict, prompt)
            logging.warning(err_msg)
            return ParsingError(err_msg)
    elif form_definition.question_includes_datatype(question_id, "boolean"):
        if outputs == "Yes":
            return True
        elif outputs == "No":
            return False
        else:
            err_msg = "Not known answer '{}' for boolean type.".format(outputs)
            logging.warning(err_msg)
            return ParsingError(err_msg)
    elif form_definition.question_includes_datatype(question_id, "number"):
        try:
            return float(outputs)
        except:
            err_msg = "'{}' should be float".format(outputs)
            logging.warning(err_msg)
            return ParsingError(err_msg)
    elif form_definition.question_includes_datatype(question_id, "integer"):
        try:
            return int(outputs)
        except:
            err_msg = "'{}' should be integer".format(outputs)
            logging.warning(err_msg)
            return ParsingError(err_msg)
    elif form_definition.question_includes_datatype(question_id, "string"):
        return outputs
    elif form_definition.question_includes_datatype(question_id, "date-time"):
        return outputs
    elif form_definition.question_includes_datatype(question_id, "date"):
        return outputs
    elif form_definition.question_includes_datatype(question_id, "time"):
        return outputs
    else:
        raise Exception("Not known question id {} field type".format(question_id))


def generate_prompt(form_definition, few_shot, question, evidences, question_id):
    if not question_id in form_definition.possible_options:
        return ""
    context = ""
    for evidence in evidences:
        context += ' '.join(evidence.split()) + "; "
    system = ""
    options = ""
    if form_definition.question_includes_datatype(question_id, "enum"):
        system = "You are given a context, a question, and a list of options ({}). Based on the context, choose the correct option".format(", ".join(string.ascii_uppercase[:len([str(op) for op in form_definition.possible_options[question_id] if op != None])]))
        for letter, option in zip(string.ascii_uppercase, [str(op) for op in form_definition.possible_options[question_id] if op != None]):
            options += "{}: {}\n".format(letter, str(option))
    elif form_definition.question_includes_datatype(question_id, "boolean"):
        system = 'You are given a context and a question. Based on the context, answer "Yes" or "No"'
    elif form_definition.question_includes_datatype(question_id, "number"):
        system = 'You are given a context and a question. Based on the context, provide the correct number as the answer. The answer may be an integer or a floating-point number'
    elif form_definition.question_includes_datatype(question_id, "integer"):
        system = 'You are given a context and a question. Based on the context, provide the correct integer as the answer'
    elif form_definition.question_includes_datatype(question_id, "string"):
        system = 'You are given a context and a question. Based on the context, provide the correct string as the answer'
    elif form_definition.question_includes_datatype(question_id, "date-time"):
        system = 'You are given a context and a question. Based on the context, provide the correct date or time in the appropriate format YYYY-MM-DDTHH:mm:ssZ'
    elif form_definition.question_includes_datatype(question_id, "date"):
        system = 'You are given a context and a question. Based on the context, provide the correct date or time in the appropriate format YYYY-MM-DD'
    elif form_definition.question_includes_datatype(question_id, "time"):
        system = 'You are given a context and a question. Based on the context, provide the correct date or time in the appropriate format HH:mm:ssZ'
    else:
        raise Exception("Not known question id {} field type".format(question_id))
    if form_definition.question_includes_datatype(question_id, "null"):
        system += ' (or answer "I don’t know" if you cannot determine the answer from the context)'
    system += '.'

    prompt = "{}\n\nQuestion: {}\n\n".format(system, question)
    if options != "":
        prompt += "Options:\n{}\n".format(options)
    prompt += few_shot
    prompt += "Context: {}\nAnswer:".format(context)
    return prompt


def generate_answer(form_definition, enumeration_value_id, question_id):
    answer = ""
    if form_definition.question_includes_datatype(question_id, "enum"):
        for letter, option in zip(string.ascii_uppercase, [str(op) for op in form_definition.possible_options[question_id] if op != None]):
            if enumeration_value_id == option:
                answer = letter
                break
        if enumeration_value_id == None:
            answer = None
        if answer == "":
            raise Exception("Enumeration value id '{}' is unknown for question '{}'".format(enumeration_value_id, question_id))
    elif form_definition.question_includes_datatype(question_id, "boolean"):
        if enumeration_value_id == True:
            answer = "Yes"
        elif enumeration_value_id == False:
            answer = "No"
        elif enumeration_value_id == None:
            answer = "I don’t know"
        else:
            raise Exception("enumeration value id {} not known for boolean type".format(enumeration_value_id))
    elif form_definition.question_includes_datatype(question_id, "number"):
        answer = enumeration_value_id
        if not (isinstance(answer, (float, int)) or answer is None):
            raise Exception("enumeration value id {} not known for number type".format(enumeration_value_id))
    elif form_definition.question_includes_datatype(question_id, "integer"):
        answer = enumeration_value_id
        if not (isinstance(answer, (int)) or answer is None):
            raise Exception("enumeration value id {} not known for integer type".format(enumeration_value_id))
    elif form_definition.question_includes_datatype(question_id, "string"):
        answer = enumeration_value_id
    elif form_definition.question_includes_datatype(question_id, "date-time"):
        answer = enumeration_value_id
    elif form_definition.question_includes_datatype(question_id, "date"):
        answer = enumeration_value_id
    elif form_definition.question_includes_datatype(question_id, "time"):
        answer = enumeration_value_id
    else:
        raise Exception("Not known question id {} field type".format(question_id))
    if answer == None:
        answer = "I don’t know"
    return answer
