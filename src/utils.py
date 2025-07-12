import requests
import pandas as pd
import smtplib
from email.mime.text import MIMEText
import os
import json
from datetime import datetime, timedelta
from config import START_YEAR, CURRENT_YEAR, API_URL
import hashlib
import hmac
import base64
import urllib.parse
import time

def fetch_historical_data():
    """
    获取2023年至今的历史开奖数据（严格符合要求）
    要求2: 使用API接口: https://history.macaumarksix.com/history/macaujc2/y/${year}
    要求3: 获取2023年至今的最新历史数据
    """
    all_data = []
    current_year = datetime.now().year
    
    print(f"开始获取历史数据: {START_YEAR}年至今...")
    
    for year in range(START_YEAR, current_year + 1):
        url = f"{API_URL}{year}"
        try:
            print(f"请求API: {url}")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            json_data = response.json()
            
            # 检查数据结构是否符合要求
            if 'data' in json_data and isinstance(json_data['data'], list):
                print(f"获取到 {year} 年数据，原始记录数: {len(json_data['data'])}")
                
                # 过滤有效数据（确保有openTime和openCode）
                valid_data = []
                for item in json_data['data']:
                    if 'openTime' in item and 'openCode' in item:
                        try:
                            # 解析日期并检查年份
                            open_time = datetime.strptime(item['openTime'], '%Y-%m-%d %H:%M:%S')
                            if open_time.year >= 2023:
                                valid_data.append(item)
                        except Exception as e:
                            print(f"解析日期错误: {e}，跳过记录: {item}")
                    else:
                        print(f"记录缺少必要字段，跳过: {item}")
                
                all_data.extend(valid_data)
                print(f"已处理 {year} 年数据，有效记录数: {len(valid_data)}")
            else:
                print(f"警告：{year}年数据格式不符合预期，跳过")
        except requests.exceptions.RequestException as e:
            print(f"网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {str(e)}")
        except Exception as e:
            print(f"获取 {year} 年数据失败: {str(e)}")
    
    if not all_data:
        print("错误：没有获取到任何有效数据")
        return pd.DataFrame()
    
    print(f"总共获取有效记录数: {len(all_data)}")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 数据清洗
    df['date'] = pd.to_datetime(df['openTime'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear  # 确保创建day_of_year列
    
    # 解析特别号码 - 取开奖号码最后一位
    try:
        df['special'] = df['openCode'].str.split(',').str[-1].astype(int)
    except Exception as e:
        print(f"解析特别号码失败: {e}")
        df['special'] = 0  # 默认值
    
    # 过滤掉2023年之前的数据（确保严格符合要求）
    df = df[df['year'] >= 2023]
    
    print(f"最终数据量: {len(df)} 条记录")
    return df.sort_values('date').reset_index(drop=True)

def zodiac_mapping(number):
    """
    将号码映射到生肖 (1-49 映射到 12 生肖)
    要求10: 只分析特别号码位置的生肖
    """
    zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
    try:
        return zodiacs[(number - 1) % 12]
    except:
        return "未知"

def send_dingtalk(message, webhook):
    """
    发送钉钉通知（支持加签验证）
    要求1: 钉钉通知功能
    """
    secret = os.getenv("DINGTALK_SECRET")
    if not secret:
        print("钉钉密钥未设置，跳过通知")
        return False
    
    print("准备发送钉钉通知...")
    
    try:
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
        
        response = requests.post(signed_webhook, json=payload, headers=headers, timeout=10)
        print(f"钉钉通知发送状态: {response.status_code}, 响应: {response.text}")
        
        if response.status_code == 200:
            print("钉钉通知发送成功")
            return True
        else:
            print(f"钉钉通知发送失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"钉钉通知发送异常: {e}")
        return False

def send_email(subject, content, receiver):
    """
    发送邮件通知
    要求1: QQ邮箱通知功能
    """
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PWD")
    
    if not sender or not password:
        print("邮箱凭据未设置，跳过邮件发送")
        return False
    
    print(f"准备发送邮件到: {receiver}...")
    
    try:
        # 创建邮件内容
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = receiver
        
        # 使用SMTP_SSL连接QQ邮箱的SMTP服务器
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()
        
        print("邮件发送成功")
        return True
    except smtplib.SMTPAuthenticationError:
        print("邮件发送失败: 认证失败，请检查邮箱和授权码")
    except smtplib.SMTPException as e:
        print(f"邮件发送失败: SMTP错误 - {e}")
    except Exception as e:
        print(f"邮件发送失败: {e}")
    
    return False

# 测试代码
if __name__ == "__main__":
    print("===== 测试 utils 模块 =====")
    
    # 测试数据获取
    print("\n测试数据获取功能...")
    test_df = fetch_historical_data()
    if not test_df.empty:
        print(f"获取到 {len(test_df)} 条记录")
        print("前5条记录:")
        print(test_df[['date', 'openCode', 'special']].head())
    else:
        print("数据获取失败")
    
    # 测试生肖映射
    print("\n测试生肖映射功能...")
    test_numbers = [1, 12, 13, 25, 37, 49]
    for num in test_numbers:
        print(f"号码 {num} -> 生肖: {zodiac_mapping(num)}")
    
    # 测试通知功能（需要设置环境变量）
    print("\n测试通知功能...")
    test_message = "这是一条测试消息\n时间: " + datetime.now().strftime('%Y-%m-%d %H:%M')
    
    if os.getenv("DINGTALK_WEBHOOK"):
        print("测试钉钉通知...")
        send_dingtalk(test_message, os.getenv("DINGTALK_WEBHOOK"))
    else:
        print("钉钉环境变量未设置，跳过测试")
    
    if os.getenv("EMAIL_USER") and os.getenv("EMAIL_PWD"):
        print("测试邮件通知...")
        send_email("测试邮件", test_message, os.getenv("EMAIL_USER"))
    else:
        print("邮件环境变量未设置，跳过测试")
    
    print("\n===== 测试完成 =====")
