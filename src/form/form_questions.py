import json
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

class FormQuestions:

    def __init__(self, question_path = "./resources/questions.json"):
        with open(question_path, 'r') as f:
            self.questions = json.load(f)


    def load_question(self, question_id, language="en", sex="masculine", nth = '1'):
        if question_id in self.questions:
            if language in self.questions[question_id]:
                if sex in self.questions[question_id][language]:
                    if len(self.questions[question_id][language]) > 0:
                        if nth in self.questions[question_id][language][sex]:
                            return self.questions[question_id][language][sex][nth]
        if question_id == "treatment.thrombolysis.drug_dose":
            return "What was the drug dose of drug used for the patient's thrombolysis?"
        logging.warning("Question ID '{}' of language '{}', sex '{}' and nth '{}' is not included in the nature question dictionary provided".format(question_id, language, sex, nth))
        return ""