import discord
import json
import os
import scripts.genai_client as genai_client
from scripts.chroma_client import ChromaDBManager
from scripts.rag_helper import RAGHelper
import re
import asyncio

channel_ids = [1473723679673417860] # whitelist channel ids

class Bot(discord.Client):
    def __init__(self, api_key, instruction=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = genai_client.GenAI(api_key, instruction)
        
        try:
            self.db_manager = ChromaDBManager(db_path="./discord_data_db")
            self.rag_helper = RAGHelper(self.db_manager)
            self.rag_enabled = True
        except Exception as e:
            print(f"⚠️ RAG initialization failed: {e}")
            self.rag_enabled = False
        
        with open("Datas/members.json", "r", encoding="utf-8") as f:
            self.members = json.load(f)
        with open("Datas/emojies.json", "r", encoding="utf-8") as f:
            self.emojies = json.load(f)

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user or message.author.bot:
            return
        if message.channel.id not in channel_ids:
            return
        
        name = ''
        for member in self.members:
            if member['id'] == message.author.id:
                name = member['name']
        
        user_query = message.content.replace(f'<@{self.user.id}>', '').strip()
        prompt = f"<@{message.author.id}> ({name}): {user_query}"
        
        
        if self.rag_enabled:
            asyncio.create_task(self._save_message_to_db(message, name))
            
            try:
                rag_data = self.rag_helper.prepare_rag_prompt(
                    query=user_query,
                    max_context_tokens=3000,
                    n_results=30,
                    include_metadata=True,
                    include_nearby_messages=True,
                    nearby_time_window_minutes=2
                )
                
                print(f"📚 Tìm thấy {rag_data['documents_used']} documents liên quan")
                
                response = self.bot.generate_with_rag(
                    user_query=prompt,
                    rag_context=rag_data['context']
                )
            except Exception as e:
                print(f"⚠️ RAG error: {e}")
                response = self.bot.generate(prompt)
        else:
            response = self.bot.generate(prompt)
        
        await message.channel.send(response)
    
    async def _save_message_to_db(self, message, author_name):
        if message.attachments:
            return
        
        try:
            content = message.content.strip()
            if not content:
                return
            
            text_without_custom_emojis = re.sub(r'<a?:\w+:\d+>', '', content)
            text_without_emojis = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text_without_custom_emojis)
            text_without_pings = re.sub(r'<@!?\d+>', '', text_without_emojis)
            
            if not text_without_pings.strip():
                return
            
            metadata = {
                "author": author_name or message.author.id,
                "timestamp": message.created_at.isoformat(),
                "message_id": str(message.id),
                "channel_id": str(message.channel.id),
                "channel_name": message.channel.name,
                "word_count": len(content.split()),
                "is_question": "?" in content,
                "message_count": 1
            }
            
            self.db_manager.add_documents(
                collection_name="messages",
                documents=[content],
                metadatas=[metadata],
                ids=[str(message.id)],
                skip_duplicates=True
            )
        except Exception as e:
            print(f"⚠️ Failed to save: {e}")