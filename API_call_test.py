# -*- coding: utf-8 -*-
# @Time : 2025/4/23 14:45
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: qwen_sft
# @File : qwen_lora_test.py
# @Software: PyCharm

import sys
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

if len(sys.argv) > 1:
    K = int(sys.argv[1])
    if len(sys.argv) > 2:
        server = sys.argv[2]
        if len(sys.argv) > 3:
            model_name = sys.argv[3]
            if len(sys.argv) > 4:
                authorization = sys.argv[4]
                if len(sys.argv) > 5:
                    num_tks = int(sys.argv[5])
                    if len(sys.argv) > 6:
                        temp = float(sys.argv[6])
                    else:
                        temp = 0
                else:
                    num_tks = 8192
                    temp = 0
            else:
                authorization = None
                num_tks = 8192
                temp = 0
        else:
            model_name = "llama3.1:8b"
            authorization = None
            num_tks = 8192
            temp = 0
    else:
        server = 'http://10.244.30.69:11434/api/chat'
        model_name = "llama3.1:8b"
        authorization = None
        num_tks = 8192
        temp = 0
else:
    K = 10
    server = 'http://10.244.30.69:11434/api/chat'
    model_name = "llama3.1:8b"
    authorization = None
    num_tks = 8192
    temp = 0


import torch
# https://api.deepseek.com/chat/completions     Bearer <DeepSeek API Key>       deepseek-chat	


import requests
# 调用ollama接口进行中文翻译
def query(text, server='http://10.244.30.69:11434/api/chat', model_name="llama3.1:8b", num_tks = 32000, temp = 0, sys_prompt=None, authorization=None):
    url = server
    headers = {'Content-Type': 'application/json'}
    if authorization:
        headers['Authorization'] = authorization
    data = {
        "model": model_name,
        "messages": None,
        "stream": False,
        "options": {
            "temperature": temp,
            "num_ctx": num_tks
        }
    }
    if not sys_prompt:
        data["messages"] = [
            {
                "role": "user",
                "content": text
            }
        ]
    else:
        data["messages"] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text}
        ]
    i = 0
    while i < 3:
        try:
            response = requests.post(url, json=data, headers=headers, timeout=300)
            break
        except requests.exceptions.RequestException:
            i += 1
            print('Request error! Sleep 10 sec ...')
            time.sleep(10)
        
    # print('response:', response.json())
    if i < 3 and response.status_code == 200:
        message = response.json()['message']['content']
        return message
    else:
        return ""

import random
import pandas as pd
from datasets import Dataset
# 加载 CSV 文件
df = pd.read_csv("traj_CN.csv")
# 将 DataFrame 转换为 Hugging Face 的 Dataset 格式
dataset = Dataset.from_pandas(df)

import pandas as pd

def find_id_by_category(file_path, category_text):
    """
    查找指定 Category 等于给定文本的记录对应 ID 列的值

    :param file_path: CSV 文件的路径
    :param category_text: 需要查找的 Category 值
    :return: 对应的 ID 列的值，返回类型为 int，如果没有找到则返回 None
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 查找 Category 列中匹配给定文本的记录
    matched_row = df[df['Category'] == category_text]

    # 如果找到匹配的记录，返回 ID 列的值，假设 ID 列是整数
    if not matched_row.empty:
        # 转换为整数并返回
        return int(matched_row['ID'].iloc[0])
    else:
        # 如果没有找到匹配的记录，返回 None
        return -1

def convert_punctuation_to_english(text):
    # 创建一个转换表，使用中文标点符号替换为英文标点符号
    # 确保中文标点符号和英文标点符号的数量相同
    translation_table = str.maketrans(
        '。？！，、：；“”（）《》【】—',  # 中文标点符号
        '.?!,,:;""()<>[]-'                # 对应的英文标点符号
    )
    # 使用translate方法进行替换
    return text.translate(translation_table)


def response_extract(text):
    try:
        # print(text)
        t_str = text.replace("车辆终点附近的POI有：\n[", "").replace("车辆终点附近的POI有:\n[", "").rstrip("。").rstrip(".").rstrip("]")
        t_str = convert_punctuation_to_english(t_str)
        # 正则表达式：提取店名和符合（大类，中类，小类）格式的括号内容
        tmp_str_lst = t_str.replace(" ", "").split('),')
        # print(tmp_str_lst)
    
        # 构造字典
        store_dict = {}
        category_dict = {}
        subcategory_dict = {}
        subsubcategory_dict = {}
    
        # 按照顺序填充字典
        for idx, text in enumerate(tmp_str_lst):
            match = text.split(',')
            # print('match:', match)
            if ')(' in match[0]:
                tmp_lst = match[0].split(')(')
                large_category = tmp_lst[1]
                store_dict[idx] = tmp_lst[0] + ')'
            else:
                tmp_lst = match[0].split('(')
                large_category = tmp_lst[1]
                store_dict[idx] = tmp_lst[0]
    
            category_dict[idx] = find_id_by_category('category/large_category_with_id.csv', large_category)
            subcategory_dict[idx] = find_id_by_category('category/medium_category_with_id.csv', match[-2])
            if match[-1][-1] == ')':
                small_category = match[-1][:-1]
            else:
                small_category = match[-1]
            subsubcategory_dict[idx] = find_id_by_category('category/small_category_with_id.csv', small_category)
    
            # print(store_dict[idx], 'aaaa', match[-3], match[-2], match[-1])
        return store_dict, category_dict, subcategory_dict, subsubcategory_dict
    except:
        return {0:-1}, {0:-1}, {0:-1}, {0:-1}


def add_records_if_needed(data_dict, y, pred=False):
    """
    如果字典中的记录数小于y，则按id顺序增加记录。

    :param data_dict: 输入的字典，格式为 {id: value}
    :param y: 目标记录数
    :return: 修改后的字典
    """
    # 获取当前字典中最大的id（如果字典为空，id从0开始）
    current_max_id = max(data_dict.keys(), default=-1)

    # 如果当前记录数小于y，则按id顺序增加记录
    while len(data_dict) < y:
        current_max_id += 1
        if pred:
            data_dict[current_max_id] = -1
        else:
            data_dict[current_max_id] = 0

    return data_dict


def hit_ratio_at_k(A, B, k=5):
    """
    计算Hit Ratio@k
    :param A: 目标字典A
    :param B: 推荐字典B
    :param k: 前k个结果
    :return: Hit Ratio@k
    """
    # 取字典B的前k个元素的键
    top_k_b = list(B.values())[:k]

    # 计算前k个结果中A中有多少个在B中出现
    hits = sum(1 for val in top_k_b if val in A.values())

    return hits / k


import numpy as np


def ndcg_at_k(A, B, k=5):
    """
    计算NDCG@k
    :param A: 目标字典A
    :param B: 推荐字典B
    :param k: 前k个结果
    :return: NDCG@k
    """
    # 取字典B的前k个元素的键
    top_k_b = list(B.values())[:k]

    # 计算A中每个值在B中的排名，0表示没有命中
    relevance = [1 if val in A.values() else 0 for val in top_k_b]

    # 计算DCG@k
    dcg = sum(relevance[i] / np.log2(i + 2) for i in range(k))

    # 计算IDCG@k (理想的DCG，假设A的相关性是最优的)
    ideal_relevance = sorted([1 if val in A.values() else 0 for val in top_k_b], reverse=True)
    idcg = sum(ideal_relevance[i] / np.log2(i + 2) for i in range(k))

    # 计算NDCG@k
    return dcg / idcg if idcg > 0 else 0

# 假设您已经有训练数据集 "dataset"

lst = []
# sample = random.choice(dataset)  # 随机选择一个样本
if os.path.exists(f'{model_name}_results_top{K}_tmp.csv'):
    df = pd.read_csv(f'{model_name}_results_top{K}_tmp.csv')
    lst = df.to_dict(orient='records')

cnt = start
for sample in dataset:
    print('@@@@\t'+str(cnt)+'\t@@@@')
    prompt = sample['prompt']
    true_label = sample['label']
    
    print_str = prompt.replace('\n', '\t')
    print(f"\tPrompt:\t{print_str}")
    print_str = true_label.replace('\n', '\t')
    print(f"\tTrue Label:\t{print_str}")
    
    response = query(prompt, server, model_name, num_tks, temp, sys_prompt=f"请根据用户输入的车辆出发时间、起点附近POI以及途经POI，按必须如下格式预测车辆终点附近至少{K}个POI:“车辆终点附近的POI有：\n[<名称>（大类，中类，小类），...]”", authorization=authorization)
    
    print_str = response.replace('\n', '\t')
    print(f"Generated Text:\t{response}")
    
    res_lst = response.split('\n\n')
    response = ""
    for txt in res_lst:
        if '车辆终点附近的POI有' in txt:
            response = txt
            break

    print_str = response.replace('\n', '\t')
    print(f"\tProcessed Text:\t{print_str}")
    
    if response == "":
        continue

    pred_store_dict, pred_category_dict, pred_subcategory_dict, pred_subsubcategory_dict = response_extract(response)

    pred_store_dict = add_records_if_needed(pred_store_dict, K, True)
    pred_category_dict = add_records_if_needed(pred_category_dict, K, True)
    pred_subcategory_dict = add_records_if_needed(pred_subcategory_dict, K, True)
    pred_subsubcategory_dict = add_records_if_needed(pred_subsubcategory_dict, K, True)

    # 显示结果
    print("\t预测店名字典:", pred_store_dict)
    print("\t预测大类字典:", pred_category_dict)
    print("\t预测中类字典:", pred_subcategory_dict)
    print("\t预测小类字典:", pred_subsubcategory_dict)

    store_dict, category_dict, subcategory_dict, subsubcategory_dict = response_extract(true_label)

    store_dict = add_records_if_needed(store_dict, K)
    category_dict = add_records_if_needed(category_dict, K)
    subcategory_dict = add_records_if_needed(subcategory_dict, K)
    subsubcategory_dict = add_records_if_needed(subsubcategory_dict, K)

    # 显示结果
    print("\t店名字典:", store_dict)
    print("\t大类字典:", category_dict)
    print("\t中类字典:", subcategory_dict)
    print("\t小类字典:", subsubcategory_dict)

    lhr = hit_ratio_at_k(pred_category_dict, category_dict, k=K)
    lndcg = ndcg_at_k(pred_category_dict, category_dict, k=K)
    print(f"\tLarge Category: Hit Ratio@{K}: {lhr:.4f}, NDCG@{K}: {lndcg:.4f}")

    mhr = hit_ratio_at_k(pred_subcategory_dict, subcategory_dict, k=K)
    mndcg = ndcg_at_k(pred_subcategory_dict, subcategory_dict, k=K)
    print(f"\tMedium Category: Hit Ratio@{K}: {mhr:.4f}, NDCG@{K}: {mndcg:.4f}")

    shr = hit_ratio_at_k(pred_subsubcategory_dict, subsubcategory_dict, k=K)
    sndcg = ndcg_at_k(pred_subsubcategory_dict, subsubcategory_dict, k=K)
    print(f"\tSmall Category: Hit Ratio@{K}: {shr:.4f}, NDCG@{K}: {sndcg:.4f}")

    dt = {'cnt': cnt, f'l_hr@{K}': lhr, f'm_hr@{K}': mhr, f's_hr@{K}': shr, f'l_ndcg@{K}': lndcg, f'm_ndcg@{K}': mndcg, f's_ndcg@{K}': sndcg}
    lst.append(dt)
    cnt += 1
    if cnt % 100 == 0:
        # 将字典列表转换为 DataFrame
        df = pd.DataFrame(lst)
        # 保存为 CSV 文件
        df.to_csv(f'{model_name}_results_top{K}_tmp.csv', index=False)
        print("Tmp_CSV文件已保存！")
        
# 将字典列表转换为 DataFrame
df = pd.DataFrame(lst)
# 保存为 CSV 文件
df.to_csv(f'{model_name}_results_top{K}.csv', index=False)
print("final_CSV文件已保存！")