import base64
import os
from google import genai
from google.genai import types

class GenAI():
    def __init__(self, api_key, instruction):
        self.generate_content_config = types.GenerateContentConfig(
            system_instruction=instruction
        )
        self.api_key = api_key
        self.instruction = instruction
        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(model="gemma-4-31b-it", config=self.generate_content_config)

    def generate(self, prompt):
        response = self.chat.send_message(prompt)
        return response.text
    
    def generate_with_rag(self, user_query: str, rag_context: str):
        rag_prompt = f"""Dựa trên ngữ cảnh sau từ lịch sử chat trong server Discord:

{rag_context}

---

Câu hỏi từ user: {user_query}

Hãy trả lời dựa trên thông tin trong ngữ cảnh trên. Nếu không tìm thấy thông tin liên quan, hãy trả lời tự nhiên dựa trên kiến thức của bạn."""
        
        response = self.chat.send_message(rag_prompt)
        return response.text
    
    def reset_chat(self):
        self.chat = self.client.chats.create(model="gemini-2.5-pro", config=self.generate_content_config)
        self.chat = self.client.chats.create(
            model="gemini-2.5-pro", 
            config=self.generate_content_config
        )
