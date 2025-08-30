from llama_index.llms.ollama import Ollama
    
class OllamaModel:
    def __init__(self):
        self.llm = Ollama(
            model="llama3.1:8b",
            request_timeout=120.0,
            temperature=0.7,
            additional_kwargs={
                "num_ctx": 2048  # or 4096
            }
        )
    
    def inference(self, prompt_text):
        print("Prompt: " + prompt_text)
        return self.llm.complete(prompt_text).text