import torch

class BertModel():
    """BERT-based model wrapper for sentiment analysis"""

    def __init__(self, model, tokenizer, id2label=None, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = model.device

        # Set label mapping
        self.id2label = id2label or getattr(model.config, "id2label", {2: "positive", 0: "negative", 1: "neutral"})

    def generate(self, prompt: str):
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        # Move to model's device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = torch.argmax(logits, dim=1).item()

        # Get label string
        if pred_idx in self.id2label:
            predicted_label = self.id2label[pred_idx]
        elif str(pred_idx) in self.id2label:
            predicted_label = self.id2label[str(pred_idx)]
        else:
            predicted_label = str(pred_idx)

        # Create a dict with full results
        result = {
            "label": predicted_label,
            "probabilities": {self.id2label[i] if i in self.id2label else (self.id2label[str(i)] if str(i) in self.id2label else str(i)):
                             float(prob) for i, prob in enumerate(probabilities)}
        }
        return result