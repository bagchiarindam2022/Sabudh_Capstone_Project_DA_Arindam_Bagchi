import requests

class InvestmentChatbot:
    def __init__(self, model="llama3.1"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

    def ask(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No answer received.")
        except Exception as e:
            return f"Chatbot unavailable. Error: {str(e)}"