
import base64
import requests
from io import BytesIO

import time

import json



meta_prompt = '''You are an annotator'''    


def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prepare_inputs(message, image, metaprompt=meta_prompt):

    base64_image = encode_image_from_pil(image)
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": 
        [
            {
                "role": "system",
                "content": [
                    metaprompt
                ]
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": message,
                    },
                    {
                        "type": "image_url",
                        "image_url": 
                        {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 800
    }
    return payload

def request_gpt4v(message, image, api_key, metaprompt=meta_prompt):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
    payload = prepare_inputs(message, image, metaprompt)
    i = 0
    while True:
        i += 1
        if i>3:
            break
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload,timeout=(10,60))
            response.raise_for_status()  
            break
        except requests.exceptions.Timeout:
            print('Time out, wait 1s & retry')
            time.sleep(1)
            continue
        except Exception as e:
            print('wait 1s & retry')
            time.sleep(1)
            continue
    if i == 4:
        raise Exception(f"Failed to make the request.")
    res = response.json()['choices'][0]['message']['content']
    return res
with open("instructions/general.json") as f:
    general_instructions = json.load(f)
def ConceptSearch(concept, api_key, ke=general_instructions['knowledge extraction']):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
    steps = ke.format(concept=concept)
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
        {
            "role": "system",
            "content": "You are an AI assitant offering definitions of various concepts"
        },
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text":steps
            }
            ]
        }
        ],
        "max_tokens": 800
    }
    i = 0
    while True:
        i += 1
        if i>3:
            break
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload,timeout=(10,60))
            response.raise_for_status()  
            break
        except requests.exceptions.Timeout:
            print('Time out, wait 1s & retry')
            time.sleep(1)
            continue
        except Exception as e:
            print('wait 1s & retry')
            time.sleep(1)
            continue
    if i == 4:
        raise Exception(f"Failed to make the request.")
    res = response.json()['choices'][0]['message']['content']
    return res

