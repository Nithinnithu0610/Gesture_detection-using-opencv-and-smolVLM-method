import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

class SmolVLMQuery:
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-500M-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def query(self, frame, prompt: str):
        # Make sure there is EXACTLY one <image> token in the prompt
        img = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        inputs = self.processor(
            text=[prompt],
            images=[img],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=50)

        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return answer
