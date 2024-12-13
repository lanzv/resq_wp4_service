import asyncio
import os
import logging
import re
import shutil
import tempfile
import time
import uuid
from src.models.answer_prediction.answer_prediction import AnswerPredictionModel
from src.models.evidence_extraction.evidence_extraction import EvidenceExtractionModel
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union


# Models for Evidence Extraction
class EvidenceExtractionRequest(BaseModel):
    context: str
    question_ids: List[str]
    language: str

class EvidenceAnswer(BaseModel):
    answer_start: Union[int, List[int]]
    text: Union[str, List[str]]
    answer_type: str
    prediction_confidence: Union[float, List[float]]

class EvidencePrediction(BaseModel):
    question_id: str
    question_confidence: float
    question_frequency: float
    answers: List[EvidenceAnswer]

class EvidenceExtractionResponse(BaseModel):
    predictions: List[EvidencePrediction]

# Models for Answer Prediction
class QuestionEvidence(BaseModel):
    question_id: str
    evidences: List[str]

class AnswerPredictionRequest(BaseModel):
    questions: List[QuestionEvidence]
    language: str

class AnswerPrediction(BaseModel):
    question_id: str
    question_confidence: float
    question_frequency: float
    enumeration_value_id: Union[None, str, int, bool]
    prediction_confidence: float

class AnswerPredictionResponse(BaseModel):
    predictions: List[AnswerPrediction]


# Initialize the FastAPI app
app = FastAPI(
    root_path="/dimbu",
    title="RES-Q+ WP4 NLP Service API",
    description="API for Evidence Extraction and Answer Prediction of RES-Q Form in Clinical Reports.",
    version="0.0.1",
)


logging.info("Loading NLP models ...")
try:
    ee_model = EvidenceExtractionModel(model_name = "/opt/saved_models/evidence_extraction")
    logging.info("Evidence Extraction model was successfully loaded!")
    ap_model = AnswerPredictionModel("/opt/saved_models/answer_prediction")
    logging.info("Answer Prediction model was successfully loaded!")
except Exception as e:
    logging.error(f"Error loading MT model: {repr(e)}")
    raise e




# API Endpoints
@app.post("/api/wp4/extract_evidences", response_model=EvidenceExtractionResponse)
def extract_evidences(request: EvidenceExtractionRequest):
    try:
        logging.info(request.dict())
        return ee_model.predict(request.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/wp4/predict_answers", response_model=AnswerPredictionResponse)
def predict_answers(request: AnswerPredictionRequest):
    try:
        logging.info(request.dict())
        return ap_model.predict(request.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
