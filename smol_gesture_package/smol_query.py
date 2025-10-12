import base64
import requests
import cv2

class SmolVLMQuery:
    def __init__(self, endpoint=None):
        """
        endpoint: your hosted SmolVLM API, e.g.,
        "https://your-hosted-smolvlm-api.com/v1"
        If None, SmolVLM queries are skipped.
        """
        self.endpoint = endpoint

    def encode_image(self, image):
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def query_batch(self, crops, prompt):
        if not self.endpoint:
            return ["Unknown"] * len(crops)

        results = []
        for img in crops:
            try:
                img_b64 = self.encode_image(img)
                payload = {
                    "model": "smolvlm",
                    "messages": [
                        {"role": "system", "content": "You are a gesture classifier. Only answer with one label."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                        ]}
                    ],
                    "temperature": 0,
                    "max_tokens": 5
                }

                r = requests.post(f"{self.endpoint}/chat/completions", json=payload, timeout=15)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"].strip()

                valid = ["Thumbs Up", "Thumbs Down",
                         "1 Finger", "2 Fingers", "3 Fingers",
                         "4 Fingers", "5 Fingers", "Unknown"]

                match = next((v for v in valid if v.lower() in text.lower()), "Unknown")
                results.append(match)
            except Exception as e:
                print("SmolVLM error:", e)
                results.append("Unknown")
        return results
