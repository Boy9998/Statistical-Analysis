import requests
import pandas as pd
import smtplib
from email.mime.text import MIMEText
import os
import json
from datetime import datetime, timedelta
from config import START_YEAR, CURRENT_YEAR, API_URL

def fetch_historical_data():
    """获取2023年至今的历史开奖数据（严格符合要求）"""
    all_data = []
    current_year = datetime.now().year
    
    for year in range(START_YEAR, current_year + 1):
        url = f"{API_URL}{year}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            json_data = response.json()
            
            if 'data' in json_data and isinstance(json_data['data'], list):
                valid_data = [item for item in json_data['data'] 
                             if 'openTime' in item and 'openCode' in item]
                
                year_data = [item for item in valid_data 
                            if datetime.strptime(item['openTime'], '%Y-%m-%d %H:%M:%S').year >= 2023]
                
                all_data.extend(year_data)
                print(f"已获取 {year} 年数据，共 {len(year_data)} 期")
        except Exception as e:
            print(f"获取 {year} 年数据失败: {str(e)}")
    
    if not all_data:
        print("错误：没有获取到任何有效数据")
        return pd.DataFrame()
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 数据清洗
    df['date'] = pd.to_datetime(df['openTime'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # 确保创建 day_of_year 列
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # 解析特别号码
    df['special'] = df['openCode'].str.split(',').str[-1].astype(int)
    
    # 过滤掉2023年之前的数据
    df = df[df['year'] >= 2023]
    
    return df.sort_values('date').reset_index(drop=True)

# 其余函数保持不变...
