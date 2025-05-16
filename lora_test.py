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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

if len(sys.argv) > 1:
    K = int(sys.argv[1])
else:
    K = 10

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# # 加载分词器
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load the base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)

# Load LoRA configuration and adapter from the saved directory
lora_config = LoraConfig.from_pretrained('./output016')  # Load from the ./output directory where LoRA is saved

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model = model.to(device)

import random
import pandas as pd
from datasets import Dataset
# 加载 CSV 文件
# load CSV
df = pd.read_csv("traj_CN.csv")
# 将 DataFrame 转换为 Hugging Face 的 Dataset 格式
# Covert DF to huggingface datset
dataset = Dataset.from_pandas(df)

# 定义预处理函数
# Define preprocess func
def preprocess_function(examples):
    inputs = [prompt for prompt in examples["prompt"]]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=512)
    labels = tokenizer(examples["label"], padding=True, truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 应用预处理函数
# Apply preprocess func
dataset = dataset.map(preprocess_function, batched=True)

import pandas as pd

def find_id_by_category(file_path, category_text):
    """
    查找指定 Category 等于给定文本的记录对应 ID 列的值
    Find the value of the ID column corresponding to the record whose specified Category is equal to the given text

    :param file_path: CSV 文件的路径 CSV Path
    :param category_text: 需要查找的 Category 值 Category for Find
    :return: 对应的 ID 列的值，返回类型为 int，如果没有找到则返回 None / corresponding int ID or None
    """
    # 读取 CSV 文件
    # Read CSV
    df = pd.read_csv(file_path)

    # 查找 Category 列中匹配给定文本的记录
    # Find match records in the Category column
    matched_row = df[df['Category'] == category_text]

    # 如果找到匹配的记录，返回 ID 列的值，假设 ID 列是整数
    # If found, return the value of the ID column, assuming int
    if not matched_row.empty:
        # 转换为整数并返回
        # Convert to int and return
        return int(matched_row['ID'].iloc[0])
    else:
        # 如果没有找到匹配的记录，返回 None
        # If no matching record, return None
        return -1

def convert_punctuation_to_english(text):
    # 创建一个转换表，使用中文标点符号替换为英文标点符号
    # Create a conversion table to replace Chinese punctuation with English punctuation
    # 确保中文标点符号和英文标点符号的数量相同
    # Make sure the number of Chinese punctuation and English punctuation is the same
    translation_table = str.maketrans(
        '。？！，、：；“”（）《》【】—',  # 中文标点符号 Chinese punctuation
        '.?!,,:;""()<>[]-'                # 对应的英文标点符号 English punctuation
    )
    # 使用translate方法进行替换
    # Replace by translate
    return text.translate(translation_table)


def response_extract(text):
    try:
        # print(text)
        t_str = text.replace("车辆终点附近的POI有：\n[", "").replace("车辆终点附近的POI有:\n[", "").rstrip("。").rstrip(".").rstrip("]")
        t_str = convert_punctuation_to_english(t_str)
        # 正则表达式：提取店名和符合（大类，中类，小类）格式的括号内容
        # Regular expression: extract the  name and the bracket content that conforms to the format
        tmp_str_lst = t_str.replace(" ", "").split('),')
        # print(tmp_str_lst)
    
        # 构造字典
        # Build dictionary
        store_dict = {}
        category_dict = {}
        subcategory_dict = {}
        subsubcategory_dict = {}
    
        # 按照顺序填充字典
        # Fill the dictionary in order
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
    If the number of records in the dictionary is less than y, then add the records in order of id.

    :param data_dict: 输入的字典，格式为 {id: value} Input Dict
    :param y: 目标记录数 Target Num of Records
    :return: 修改后的字典 Modified Dict
    """
    # 获取当前字典中最大的id（如果字典为空，id从0开始）
    # Get the largest id in the current dictionary (if the dictionary is empty, the id starts at 0)
    current_max_id = max(data_dict.keys(), default=-1)

    # 如果当前记录数小于y，则按id顺序增加记录
    # If the current number of records is less than y, add records in order of id
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
    Calculate Hit Ratio@k
    :param A: 目标字典A target dictionary A
    :param B: 推荐字典B recommended dictionary B
    :param k: 前k个结果 top k results
    :return: Hit Ratio@k
    """
    # 取字典B的前k个元素的键
    # Get the keys of the first k elements of dictionary B
    top_k_b = list(B.values())[:k]

    # 计算前k个结果中A中有多少个在B中出现
    # Calculate how many of A appear in B among the first k results
    hits = sum(1 for val in top_k_b if val in A.values())

    return hits / k


import numpy as np


def ndcg_at_k(A, B, k=5):
    """
    计算NDCG@k
    Calculate NDCG@k
    :param A: 目标字典A target dictionary A
    :param B: 推荐字典B recommended dictionary B
    :param k: 前k个结果 top k results
    :return: NDCG@k
    """
    # 取字典B的前k个元素的键
    # Get the keys of the first k elements of dictionary B
    top_k_b = list(B.values())[:k]

    # 计算A中每个值在B中的排名，0表示没有命中
    # Calculate the ranking of each value in A in B, 0 means no hit
    relevance = [1 if val in A.values() else 0 for val in top_k_b]

    # 计算DCG@k
    # Calc DCG@k
    dcg = sum(relevance[i] / np.log2(i + 2) for i in range(k))

    # 计算IDCG@k (理想的DCG，假设A的相关性是最优的)
    # Calculate IDCG@k (ideal DCG, assuming that the correlation of A is optimal)
    ideal_relevance = sorted([1 if val in A.values() else 0 for val in top_k_b], reverse=True)
    idcg = sum(ideal_relevance[i] / np.log2(i + 2) for i in range(k))

    # 计算NDCG@k
    # Calc NGCG@K
    return dcg / idcg if idcg > 0 else 0

# 假设您已经有训练数据集 "dataset"
# Assuming you already have a training dataset "dataset"
# sample = random.choice(dataset)  # 随机选择一个样本 Randomly select a sample

lst = []
cnt = 1
for sample in dataset:
    print('@@@@\t'+str(cnt)+'\t@@@@')
    prompt = sample['prompt']
    true_label = sample['label']

    print_str = prompt.replace('\n', '\t')
    print(f"\tPrompt:\t{print_str}")
    print_str = true_label.replace('\n', '\t')
    print(f"\tTrue Label:\t{print_str}")

    messages = [
        {"role": "system", "content": f"请根据用户输入的车辆出发时间、起点附近POI以及途经POI，按如下格式预测车辆终点附近至少{K}个POI:“车辆终点附近的POI有：\n[<名称>(大类,中类,小类), ...]”"},
        # Please predict at least {K} POIs near the vehicle's destination according to the vehicle's departure time, POIs near the starting point, and POIs along the way, in the following format: "The POIs near the vehicle's destination are: \n[<name>(large category, medium category, small category), ...]
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = model_inputs.to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print_str = response.replace("\n", "\t")
    print(f"\tGenerated Text:\t{print_str}")

    pred_store_dict, pred_category_dict, pred_subcategory_dict, pred_subsubcategory_dict = response_extract(response)

    pred_store_dict = add_records_if_needed(pred_store_dict, K, True)
    pred_category_dict = add_records_if_needed(pred_category_dict, K, True)
    pred_subcategory_dict = add_records_if_needed(pred_subcategory_dict, K, True)
    pred_subsubcategory_dict = add_records_if_needed(pred_subsubcategory_dict, K, True)

    # 显示结果
    # Show result
    print("\t预测店名字典:", pred_store_dict) # pred of name
    print("\t预测大类字典:", pred_category_dict) # pred of category
    print("\t预测中类字典:", pred_subcategory_dict) # pred of subcategory
    print("\t预测小类字典:", pred_subsubcategory_dict) # pred of subsubcategory

    store_dict, category_dict, subcategory_dict, subsubcategory_dict = response_extract(true_label)

    store_dict = add_records_if_needed(store_dict, K)
    category_dict = add_records_if_needed(category_dict, K)
    subcategory_dict = add_records_if_needed(subcategory_dict, K)
    subsubcategory_dict = add_records_if_needed(subsubcategory_dict, K)

    # 显示结果
    # Show result
    print("\t店名字典:", store_dict) # dict of name
    print("\t大类字典:", category_dict) # dict of category
    print("\t中类字典:", subcategory_dict) # dict of subcategory
    print("\t小类字典:", subsubcategory_dict) # dict of subsubcategory

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
        # Convert list of dict to DataFrame
        df = pd.DataFrame(lst)
        # 保存为 CSV 文件
        # Save to CSV
        df.to_csv(f'{model_name}_results_top{K}_tmp.csv', index=False)
        print("Tmp_CSV文件已保存！")
        
# 将字典列表转换为 DataFrame
# Convert list of dict to DataFrame
df = pd.DataFrame(lst)
# 保存为 CSV 文件
# Save to CSV
df.to_csv(f'{model_name}_results_top{K}.csv', index=False)
print("final_CSV文件已保存！")