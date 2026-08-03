import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
if not api_key:
    raise ValueError("未找到 ALPHA_VANTAGE_API_KEY，请检查 .env 文件。")

params = {
    "function": "TIME_SERIES_DAILY",
    "symbol": "JPM",
    "outputsize": "compact",
    "apikey": api_key,
}
url = "https://www.alphavantage.co/query?" + urlencode(params)

try:
    with urlopen(url, timeout=60) as response:
        response_text = response.read().decode("utf-8")
except Exception:
    raise RuntimeError("无法连接 Alpha Vantage，请检查网络后重试。")

data = json.loads(response_text)

if "Error Message" in data:
    raise RuntimeError("Alpha Vantage 返回了接口错误。")

if "Information" in data:
    raise RuntimeError("Alpha Vantage 当前请求额度受限，请稍后再试。")

series = data.get("Time Series (Daily)")
if not series:
    raise RuntimeError("未收到预期的日线数据。")

print("Alpha Vantage API 测试成功。")
print(f"标的：{data['Meta Data']['2. Symbol']}")
print(f"返回日线记录数：{len(series)}")