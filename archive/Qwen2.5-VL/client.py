import requests

def request_OneTextMultiImage(api_url, image_urls, prompt, model="Qwen2.5-VL-3B-Instruct"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        }
    ]

    for image_url in image_urls:
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        )

    payload = {
        "model": model,
        "messages": messages
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

ask_about_images = request_OneTextMultiImage
