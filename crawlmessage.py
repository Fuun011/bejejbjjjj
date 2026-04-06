
import discord
from discord import app_commands
import os
import dotenv
import logging
from typing import List, Dict
from scripts.chroma_client import ChromaDBManager
from scripts.clean import process_log_file, DEFAULT_CONFIG
dotenv.load_dotenv()
DISCORD_KEY = os.getenv("DISCORD_KEY")
if not DISCORD_KEY:
    raise ValueError("DISCORD_KEY không được tìm thấy trong file .env")
gateway_logger = logging.getLogger('discord.gateway')
gateway_logger.setLevel(logging.ERROR)
try:
    db_manager = ChromaDBManager(db_path="./discord_data_db")
except ValueError as e:
    print(f"LỖI KHỞI TẠO: {e}")
    exit()
_MESSAGES_PER_FETCH = 100
_PROCESS_BATCH_SIZE = 50
class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def on_ready(self):
        await self.tree.sync()
        print(f'Logged in as {self.user}. Bot is ready.')
        print('Sử dụng lệnh /sync_channel trong kênh Discord để bắt đầu.')
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = MyClient(intents=intents)

async def _process_and_save_batch(message_batch: List[discord.Message], db_manager: ChromaDBManager, channel: discord.TextChannel) -> int:
    """
    Xử lý một batch tin nhắn và lưu vào database.
    
    Args:
        message_batch: Danh sách các tin nhắn Discord
        db_manager: Đối tượng quản lý ChromaDB
        channel: Discord channel object để lấy metadata
        
    Returns:
        Số lượng documents đã thêm vào DB
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
        
        added_count = db_manager.add_documents(
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

@client.tree.command(name="clear_db", description="[ADMIN ONLY] Xóa toàn bộ dữ liệu trong VectorDB.")
async def clear_db(interaction: discord.Interaction):
    """
    Lệnh xóa toàn bộ database. Chỉ user ID 749178876776677396 được dùng.
    """
    ADMIN_USER_ID = 749178876776677396
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message(
            "❌ Bạn không có quyền sử dụng lệnh này!",
            ephemeral=True
        )
        return
    
    await interaction.response.defer(thinking=True, ephemeral=True)
    
    try:
        stats = db_manager.get_collection_stats("messages")
        doc_count = stats.get('count', 0)
        
        success = db_manager.delete_collection("messages")
        
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

@client.tree.command(name="sync_channel", description="[ADMIN ONLY] Lấy lịch sử và nạp vào VectorDB cho kênh này.")
@app_commands.describe(max_messages="Số lượng tin nhắn tối đa muốn lấy (ví dụ: 5000).")
async def sync_channel(interaction: discord.Interaction, max_messages: int = 3000):
    """
    Lệnh này sẽ lấy lịch sử chat, dọn dẹp, và lưu vào ChromaDB.
    """
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
                    added = await _process_and_save_batch(message_buffer, db_manager, channel)
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
        added = await _process_and_save_batch(message_buffer, db_manager, channel)
        total_added_to_db += added
        message_buffer.clear()

    final_message = f"✅ Hoàn tất! Đã xử lý {total_processed_count} tin nhắn và thêm tổng cộng {total_added_to_db} tài liệu vào cơ sở dữ liệu."
    print(final_message)
    await interaction.followup.send(final_message, ephemeral=True)

@client.tree.command(name="continue_sync", description="[ADMIN ONLY] Tiếp tục crawl từ message cũ nhất trong DB.")
@app_commands.describe(max_messages="Số lượng tin nhắn tối đa muốn lấy thêm (ví dụ: 3000).")
async def continue_sync(interaction: discord.Interaction, max_messages: int = 3000):
    """
    Lệnh này sẽ tìm message cũ nhất trong DB và crawl tiếp từ đó về quá khứ.
    """
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
        collection = db_manager.get_or_create_collection("messages")
        
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
                    added = await _process_and_save_batch(message_buffer, db_manager, channel)
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
        added = await _process_and_save_batch(message_buffer, db_manager, channel)
        total_added_to_db += added
        message_buffer.clear()
    
    final_message = f"✅ Hoàn tất continue sync! Đã xử lý {total_processed_count} tin nhắn và thêm {total_added_to_db} tài liệu mới vào DB."
    print(final_message)
    await interaction.followup.send(final_message, ephemeral=True)


if __name__ == "__main__":
    client.run(DISCORD_KEY)