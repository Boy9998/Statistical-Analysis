import requests
import pandas as pd
import numpy as np
from config import START_YEAR, CURRENT_YEAR, API_URL
import smtplib
from email.mime.text import MIMEText
import os
import json
from datetime import datetime

def fetch_historical_data():
    """获取2023年至今的历史开奖数据"""
    all_data = []
    for year in range(START_YEAR, int(CURRENT_YEAR) + 1):
        url = f"{API_URL}{year}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            year_data = response.json()
            all_data.extend(year_data)
            print(f"已获取 {year} 年数据，共 {len(year_data)} 期")
        except Exception as e:
            print(f"获取 {year} 年数据失败: {str(e)}")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 数据清洗
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # 解析特别号码
    df['special'] = df['num'].str.split('+').str[1].astype(int)
    
    return df.sort_values('date').reset_index(drop=True)

def zodiac_mapping(number):
    """将号码映射到生肖 (1-49 映射到 12 生肖)"""
    zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
    return zodiacs[(number - 1) % 12]

def send_dingtalk(message, webhook):
    """发送钉钉通知（支持加签验证）"""
    import hashlib
    import hmac
    import base64
    import urllib.parse
    import time
    
    secret = os.getenv("DINGTALK_SECRET")
    if not secret:
        print("钉钉密钥未设置，跳过通知")
        return False
    
    # 1. 生成时间戳和签名
    timestamp = str(round(time.time() * 1000))
    secret_enc = secret.encode('utf-8')
    string_to_sign = f'{timestamp}\n{secret}'
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    
    # 2. 构建带签名的URL
    signed_webhook = f"{webhook}&timestamp={timestamp}&sign={sign}"
    
    # 3. 发送请求
    headers = {"Content-Type": "application/json"}
    payload = {
        "msgtype": "text",
        "text": {"content": f"彩票分析报告:\n{message}"}
    }
    try:
        response = requests.post(signed_webhook, json=payload, headers=headers)
        print(f"钉钉通知发送状态: {response.status_code}")
        return True
    except Exception as e:
        print(f"钉钉通知发送失败: {e}")
        return False
