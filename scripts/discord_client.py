import discord
from discord import app_commands
import json
import os
import logging
import scripts.genai_client as genai_client
from scripts.chroma_client import ChromaDBManager
from scripts.rag_helper import RAGHelper
from scripts.clean import process_log_file
import re
import asyncio
from typing import List

channel_ids = [1473723679673417860] # whitelist channel ids
gateway_logger = logging.getLogger('discord.gateway')
gateway_logger.setLevel(logging.ERROR)
_MESSAGES_PER_FETCH = 100
_PROCESS_BATCH_SIZE = 50

class Bot(discord.Client):
    def __init__(self, api_key, instruction=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = app_commands.CommandTree(self)
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

        self._register_crawl_commands()

    async def on_ready(self):
        try:
            await self.tree.sync()
        except Exception as e:
            print(f"⚠️ Không thể sync slash commands: {e}")
        print(f'Logged in as {self.user}')
        print('Sử dụng lệnh /sync_channel hoặc /continue_sync trong kênh Discord để crawl dữ liệu.')

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

    def _register_crawl_commands(self):
        @self.tree.command(name="clear_db", description="[ADMIN ONLY] Xóa toàn bộ dữ liệu trong VectorDB.")
        async def clear_db(interaction: discord.Interaction):
            ADMIN_USER_ID = 749178876776677396
            if interaction.user.id != ADMIN_USER_ID:
                await interaction.response.send_message(
                    "❌ Bạn không có quyền sử dụng lệnh này!",
                    ephemeral=True
                )
                return
            
            await interaction.response.defer(thinking=True, ephemeral=True)
            
            try:
                stats = self.db_manager.get_collection_stats("messages")
                doc_count = stats.get('count', 0)
                success = self.db_manager.delete_collection("messages")
                
                if success:
                    message = f"✅ Đã xóa thành công collection 'messages'!\n📊 Đã xóa {doc_count} documents."
                    print(f"INFO: Admin {interaction.user.name} đã xóa database. Documents deleted: {doc_count}")
                else:
                    message = "⚠️ Có lỗi khi xóa collection. Xem log để biết chi tiết."
                    
                await interaction.followup.send(message, ephemeral=True)
                
            except Exception as e:
                error_message = f"❌ Lỗi khi xóa database: {e}"
                print(f"ERROR: {error_message}")
                await interaction.followup.send(error_message, ephemeral=True)

        @self.tree.command(name="sync_channel", description="[ADMIN ONLY] Lấy lịch sử và nạp vào VectorDB cho kênh này.")
        @app_commands.describe(max_messages="Số lượng tin nhắn tối đa muốn lấy (ví dụ: 5000).")
        async def sync_channel(interaction: discord.Interaction, max_messages: int = 3000):
            ADMIN_USER_ID = 749178876776677396
            if interaction.user.id != ADMIN_USER_ID:
                await interaction.response.send_message(
                    "❌ Bạn không có quyền sử dụng lệnh này!",
                    ephemeral=True
                )
                return
            
            channel = interaction.channel
            
            await interaction.response.defer(thinking=True, ephemeral=True)
            
            print(f"\nBắt đầu quá trình đồng bộ cho kênh: '{channel.name}'")
            
            total_processed_count = 0
            total_added_to_db = 0
            oldest_message = None
            message_buffer = []
            
            while total_processed_count < max_messages:
                limit = min(_MESSAGES_PER_FETCH, max_messages - total_processed_count)
                if limit <= 0:
                    break
                
                try:
                    if oldest_message is None:
                        history = channel.history(limit=limit)
                    else:
                        history = channel.history(limit=limit, before=oldest_message)
                    
                    batch_count = 0
                    async for msg in history:
                        message_buffer.append(msg)
                        oldest_message = msg
                        batch_count += 1
                        total_processed_count += 1
                        
                        if len(message_buffer) >= _PROCESS_BATCH_SIZE:
                            added = await self._process_and_save_batch(message_buffer, channel)
                            total_added_to_db += added
                            message_buffer.clear()
                            
                            if total_processed_count % 100 == 0 and total_added_to_db > 0:
                                await interaction.followup.send(
                                    f"Đã xử lý {total_processed_count}/{max_messages} tin nhắn. Thêm {total_added_to_db} tài liệu.",
                                    ephemeral=True
                                )
                    
                    if batch_count == 0:
                        print("INFO: Không còn tin nhắn để lấy.")
                        break
                        
                except Exception as e:
                    print(f"ERROR: Lỗi khi lấy tin nhắn: {e}")
                    break
            
            if message_buffer:
                print(f"INFO: Đang xử lý {len(message_buffer)} tin nhắn còn lại trong buffer...")
                added = await self._process_and_save_batch(message_buffer, channel)
                total_added_to_db += added
                message_buffer.clear()

            final_message = f"✅ Hoàn tất! Đã xử lý {total_processed_count} tin nhắn và thêm tổng cộng {total_added_to_db} tài liệu vào cơ sở dữ liệu."
            print(final_message)
            await interaction.followup.send(final_message, ephemeral=True)

        @self.tree.command(name="continue_sync", description="[ADMIN ONLY] Tiếp tục crawl từ message cũ nhất trong DB.")
        @app_commands.describe(max_messages="Số lượng tin nhắn tối đa muốn lấy thêm (ví dụ: 3000).")
        async def continue_sync(interaction: discord.Interaction, max_messages: int = 3000):
            ADMIN_USER_ID = 749178876776677396
            if interaction.user.id != ADMIN_USER_ID:
                await interaction.response.send_message(
                    "❌ Bạn không có quyền sử dụng lệnh này!",
                    ephemeral=True
                )
                return
            
            channel = interaction.channel
            
            await interaction.response.defer(thinking=True, ephemeral=True)
            
            print(f"\nTìm message cũ nhất trong DB cho channel: '{channel.name}'")
            
            try:
                collection = self.db_manager.get_or_create_collection("messages")
                results = collection.get(
                    where={"channel_id": str(channel.id)},
                    include=["metadatas"]
                )
                
                if not results['ids'] or len(results['ids']) == 0:
                    await interaction.followup.send(
                        "❌ Không tìm thấy message nào trong DB cho channel này. Hãy dùng /sync_channel trước!",
                        ephemeral=True
                    )
                    return
                
                oldest_timestamp = None
                oldest_message_id = None
                
                for metadata in results['metadatas']:
                    timestamp_str = metadata.get('timestamp')
                    msg_id = metadata.get('message_id')
                    
                    if timestamp_str:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp_str)
                        
                        if oldest_timestamp is None or timestamp < oldest_timestamp:
                            oldest_timestamp = timestamp
                            oldest_message_id = msg_id
                
                if not oldest_message_id:
                    await interaction.followup.send(
                        "❌ Không thể xác định message cũ nhất trong DB.",
                        ephemeral=True
                    )
                    return
                
                print(f"INFO: Message cũ nhất trong DB: ID={oldest_message_id}, Timestamp={oldest_timestamp}")
                
                try:
                    oldest_discord_message = await channel.fetch_message(int(oldest_message_id))
                except discord.NotFound:
                    await interaction.followup.send(
                        f"⚠️ Message ID {oldest_message_id} không tồn tại trên Discord. Có thể đã bị xóa.\n"
                        "Sẽ crawl từ message cũ nhất hiện có trên channel.",
                        ephemeral=True
                    )
                    oldest_discord_message = None
                except Exception as e:
                    print(f"ERROR: Không thể fetch message: {e}")
                    oldest_discord_message = None
                
                await interaction.followup.send(
                    f"🔍 Đã tìm thấy message cũ nhất trong DB: {oldest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"🚀 Bắt đầu crawl {max_messages} tin nhắn cũ hơn...",
                    ephemeral=True
                )
                
            except Exception as e:
                error_msg = f"❌ Lỗi khi tìm message cũ nhất: {e}"
                print(error_msg)
                await interaction.followup.send(error_msg, ephemeral=True)
                return
            
            print(f"\nBắt đầu quá trình crawl tiếp tục cho kênh: '{channel.name}'")
            
            total_processed_count = 0
            total_added_to_db = 0
            message_buffer = []
            
            current_before = oldest_discord_message
            
            while total_processed_count < max_messages:
                limit = min(_MESSAGES_PER_FETCH, max_messages - total_processed_count)
                if limit <= 0:
                    break
                
                try:
                    if current_before:
                        history = channel.history(limit=limit, before=current_before)
                    else:
                        history = channel.history(limit=limit)
                    
                    batch_count = 0
                    async for msg in history:
                        message_buffer.append(msg)
                        current_before = msg
                        batch_count += 1
                        total_processed_count += 1
                        
                        if len(message_buffer) >= _PROCESS_BATCH_SIZE:
                            added = await self._process_and_save_batch(message_buffer, channel)
                            total_added_to_db += added
                            message_buffer.clear()
                            
                            if total_processed_count % 100 == 0 and total_added_to_db > 0:
                                await interaction.followup.send(
                                    f"Đã xử lý {total_processed_count}/{max_messages} tin nhắn. "
                                    f"Tổng cộng đã thêm {total_added_to_db} tài liệu mới vào DB.",
                                    ephemeral=True
                                )
                    
                    if batch_count == 0:
                        print("INFO: Không còn tin nhắn cũ hơn để lấy.")
                        break
                
                except Exception as e:
                    print(f"ERROR: Lỗi khi lấy tin nhắn: {e}")
                    break
            
            if message_buffer:
                print(f"INFO: Đang xử lý {len(message_buffer)} tin nhắn còn lại...")
                added = await self._process_and_save_batch(message_buffer, channel)
                total_added_to_db += added
                message_buffer.clear()
            
            final_message = f"✅ Hoàn tất continue sync! Đã xử lý {total_processed_count} tin nhắn và thêm {total_added_to_db} tài liệu mới vào DB."
            print(final_message)
            await interaction.followup.send(final_message, ephemeral=True)

    async def _process_and_save_batch(self, message_batch: List[discord.Message], channel: discord.TextChannel) -> int:
        """
        Xử lý một batch tin nhắn và lưu vào database.
        """
        if not message_batch:
            return 0
        
        message_id_map = {}
        log_lines = []
        for msg in reversed(message_batch):
            timestamp = msg.created_at.isoformat()
            author_id = str(msg.author.id)
            content = msg.content if msg.content else ""
            
            key = f"{timestamp}|{author_id}"
            if key not in message_id_map:
                message_id_map[key] = []
            message_id_map[key].append(msg.id)
            
            log_lines.append(f"[{timestamp}] {author_id}: {content}")
        
        temp_log_path = f"temp_batch_{id(message_batch)}.txt"
        try:
            with open(temp_log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines))
            cleaned_documents = process_log_file(temp_log_path)
        finally:
            if os.path.exists(temp_log_path):
                os.remove(temp_log_path)
        
        if not cleaned_documents:
            return 0
        
        try:
            contents = []
            metadatas = []
            ids = []
            
            author_message_pool = {}
            for key, msg_ids in message_id_map.items():
                _, author = key.split('|', 1)
                if author not in author_message_pool:
                    author_message_pool[author] = []
                author_message_pool[author].extend(msg_ids)
            
            used_message_ids = set()
            
            for doc in cleaned_documents:
                author = doc['author']
                
                message_id = None
                if author in author_message_pool:
                    for mid in author_message_pool[author]:
                        if mid not in used_message_ids:
                            message_id = mid
                            used_message_ids.add(mid)
                            break
                
                if not message_id:
                    print(f"WARNING: Không tìm thấy message_id chưa dùng cho: {author} - {doc['timestamp'][:19]}")
                    continue
                
                contents.append(doc['content'])
                metadatas.append({
                    'author': doc['author'], 
                    'timestamp': doc['timestamp'],
                    'message_id': str(message_id),
                    'word_count': doc.get('word_count', len(doc['content'].split())),
                    'is_question': doc.get('is_question', False),
                    'channel_id': str(channel.id),
                    'channel_name': channel.name
                })
                ids.append(f"msg-{message_id}")
            
            if not contents:
                print("WARNING: Không có documents hợp lệ để thêm sau khi map message_id.")
                return 0
            
            added_count = self.db_manager.add_documents(
                collection_name="messages",
                documents=contents,
                metadatas=metadatas,
                ids=ids,
                skip_duplicates=True
            )
            
            if added_count > 0:
                print(f"SUCCESS: Đã thêm {added_count} tài liệu mới vào DB.")
            
            return added_count
            
        except Exception as e:
            print(f"ERROR: Lỗi khi thêm tài liệu vào DB: {e}")
            return 0
    
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
