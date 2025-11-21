import csv
import time
from os import makedirs

import pandas as pd
from openai import OpenAI
import re
import os

# Specify the output CSV file path
output_csv = os.path.join('F:\dataSet', "deepseekchat.csv")

# Function to read Java files from a folder
def read_java_files(folder_path):
    java_content = ""
    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a Java file
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                # Open and read the file content
                with open(file_path, "r", encoding="utf-8") as f:
                    java_content += f.read() + "\n"  # Append the file content to java_content
    return java_content

deepseekData = []

# Read the data from the CSV file
data = pd.read_csv("F:\dataSet\merged_records.csv", encoding='gbk')

for _, row in data.iterrows():
    try:
        folder_path = row['file_path']

        result = read_java_files(folder_path)
        id = row['instance_id']
        print(f'Currently processing {id}')
        # Set up the API client

        client = OpenAI(api_key="", base_url="https://api.deepseek.com")
        # GPT-4o
        # client = OpenAI(api_key="", base_url="https://api.openai.com/v1")
        a_time = time.time()
        messages = [{"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user",
                     "content": f"I will send you some code next. Please analyze what design pattern it uses. Just provide the name of the design pattern without any explanation. Select only one. The possible design patterns are AbstractFactory, Adapter, Decorator, Facade, FactoryMethod, Proxy, Singleton. Here is the code: {result}"}]
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages)
        print(response.choices[0].message.content)
        deepseekData.append({
            "id": row['instance_id'],
            "true": row['Category'],
            "prediction": response.choices[0].message.content
        })
        b_time = time.time()
        c_time = b_time - a_time
        print(f'Training took {c_time:.2f} seconds')
    except Exception as e:
        print(f"An error occurred, skipping the current row: {e}")
        continue

# Specify the folder path

with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["id", "true", "prediction"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deepseekData)