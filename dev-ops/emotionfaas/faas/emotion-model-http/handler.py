
import json
import requests

def handle(event, context):

    url = "http://45.113.235.20:8080/function/emotion-model"
    data_url = event.body
    emotion_request = requests.post(url, data=data_url)

    #output = {"prediction": labels, "Confidence level" : str(round(prediction,2))}

    return {
        "statusCode": 200,
        "body": emotion_request.text,
        "headers": {
          'Content-type': 'text/plain',
          "Access-Control-Allow-Origin": "http://45.113.235.20"
        }
    }
