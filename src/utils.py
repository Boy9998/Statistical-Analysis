import requests
import pandas as pd
import smtplib
from email.mime.text import MIMEText
import os
import json
from datetime import datetime
from config import START_YEAR, CURRENT_YEAR, API_URL

def fetch_historical_data():
    """获取2023年至今的历史开奖数据"""
    all_data = []
    for year in range(START_YEAR, int(CURRENT_YEAR) + 1):
        url = f"{API_URL}{year}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            
            # 检查API返回的数据结构
            if 'data' in json_data and isinstance(json_data['data'], list):
                year_data = json_data['data']
                all_data.extend(year_data)
                print(f"已获取 {year} 年数据，共 {len(year_data)} 期")
            else:
                print(f"警告：{year}年数据格式不符合预期，跳过")
                continue
        except Exception as e:
            print(f"获取 {year} 年数据失败: {str(e)}")
    
    # 转换为DataFrame
    if not all_data:
        print("错误：没有获取到任何数据")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    # 数据清洗 - 使用API返回的实际字段名
    if 'openTime' in df.columns:
        df['date'] = pd.to_datetime(df['openTime'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
    else:
        print("警告：数据中缺少'openTime'字段")
    
    # 解析特别号码 - 使用API返回的实际字段名
    if 'openCode' in df.columns:
        # 示例：openCode = "37,30,49,16,09,12,45"
        # 我们取最后一个号码作为特别号码
        df['special'] = df['openCode'].str.split(',').str[-1].astype(int)
    else:
        print("警告：数据中缺少'openCode'字段")
    
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

def send_email(subject, content, receiver):
    """发送邮件通知"""
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PWD")
    
    if not sender or not password:
        print("邮箱凭据未设置，跳过邮件发送")
        return False
    
    # 创建邮件内容
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    
    try:
        # 使用SMTP_SSL连接QQ邮箱的SMTP服务器
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()
        print("邮件发送成功")
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False
