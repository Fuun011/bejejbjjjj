import time, os
from scripts.discord_client import *
import dotenv

dotenv.load_dotenv()
discord_key = os.getenv("DISCORD_KEY")
api_key = os.getenv("API_KEY")


with open("Datas/instruction.txt","r", encoding="utf-8") as instr:
    with open("Datas/members.json", "r", encoding="utf-8") as mem:
        with open("Datas/emojies.json", "r", encoding="utf-8") as emj:
            instruction = instr.read() + "\n\n" + "Danh sách thành viên trong server:\n" + mem.read() + "\n\n" + "Danh sách emojies trong server:\n" + emj.read()

intents = discord.Intents.all()
client = Bot(api_key=api_key, instruction=instruction, intents=intents)

while 1:
    try:
        client.run(discord_key)
    except:
        os.system("kill 1")