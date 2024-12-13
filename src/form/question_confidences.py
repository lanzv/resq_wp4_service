import json


class QuestionConfidences:
    def __init__(self, confidences_path = "./resources/confidences.json"):
        with open(confidences_path, 'r') as f:
            self.confidences = json.load(f)

    def get_evidence_extraction_confidence(self, question_id, language):
        if not language in self.confidences["evidence_extraction"]:
            return {
                "question_confidence": 0.0,
                "question_frequency": 0.0
            }
        if not question_id in self.confidences["evidence_extraction"][language]:
            return {
                "question_confidence": 0.0,
                "question_frequency": 0.0
            }
        return self.confidences["evidence_extraction"][language][question_id]

    def get_answer_prediction_confidence(self, question_id, language):
        if not language in self.confidences["answer_prediction"]:
            return {
                "question_confidence": 0.0,
                "question_frequency": 0.0
            }
        if not question_id in self.confidences["answer_prediction"][language]:
            return {
                "question_confidence": 0.0,
                "question_frequency": 0.0
            }
        return self.confidences["answer_prediction"][language][question_id]