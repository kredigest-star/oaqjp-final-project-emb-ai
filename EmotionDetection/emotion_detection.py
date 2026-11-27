"""Module for emotion detection using the Watson NLP library."""
import json

import requests


def emotion_detector(text_to_analyze):
    """Analyze the emotions of the given text using the Watson NLP library."""
    # URL of the emotion detection service
    url = ('https://sn-watson-emotion.labs.skills.network/'
           'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict')
    # Constructing the request payload in the expected format
    myobj = { "raw_document": { "text": text_to_analyze } }
    # Custom header specifying the model ID for the emotion detection service
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    # Sending a POST request to the emotion detection API
    response = requests.post(url, json=myobj, headers=header, timeout=30)
    # Parsing the response to extract the emotion scores
    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']
    # Finding the dominant emotion, which is the emotion with the highest score
    dominant_emotion = max(emotions, key=emotions.get)
    return {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }
