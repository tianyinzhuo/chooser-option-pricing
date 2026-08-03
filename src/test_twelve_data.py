import json
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TWELVE_DATA_API_KEY")
if not api_key:
    raise ValueError("未找到 TWELVE_DATA_API_KEY，请检查 .env 文件。")

url = (
    "https://api.twelvedata.com/time_series"
    "?symbol=JPM&interval=1day&outputsize=1"
    f"&apikey={api_key}"
)

result = subprocess.run(
    ["curl.exe", "-L", "--silent", "--show-error", "--max-time", "30", url],
    capture_output=True,
    text=True,
    check=True,
)

data = json.loads(result.stdout)

if data.get("status") == "error":
    raise RuntimeError(data.get("message", "Twelve Data API 返回未知错误。"))

print("Twelve Data API 测试成功。")
print(f"标的：{data['meta']['symbol']}")
print(f"交易所：{data['meta']['exchange']}")
print(f"返回记录数：{len(data['values'])}")