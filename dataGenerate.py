from os import makedirs

from openai import OpenAI
import re
import os

# Set API client
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

# Global call count counter
i = 3000

# Scene keywords
keywords = [
    "File Encryption System",
    "Computer Assembly Process",
]

abstract_factory_keywords = [
    "Cross-platform UI Component Library",
    "Game Character Creation System",
]

decorator_keywords = [
    "Enhancing User Authentication System",
    "Extending Log Recording System",
    "Enhancing Data Encryption Transmission",
    "Adding Image Processing Effects",
]

factory_mode_keywords = [
    "Production Scheduling Optimization", "Real-time Production Monitoring", "Equipment Utilization Improvement",
    "Production Bottleneck Identification", "Multi-production Line Coordination", "Real-time Quality Data Analysis",
    "Quality Improvement Plan Generation", "Quality Prediction and Prevention", "Automated Quality Standards",
    "Supplier Quality Evaluation",
]

proxy_mode_keywords = [
    "Access Permission Verification", "Resource Access Control", "Operation Log Recording", "Security Proxy",
    "Remote Resource Access", "Network Request Proxy", "API Call Interception", "Cache Management",
    "Lazy Loading", "Connection Pool Management",
]

# Extract and save Java design pattern example code
def extract_and_save_java_code(response):
    # Get the text returned by the API
    text = response.choices[0].message.content

    # Find Java code blocks (Assuming code blocks start with "START_JAVA_CODE" and end with "END_JAVA_CODE")
    java_code_blocks = re.findall(r"START_JAVA_CODE(.*?)END_JAVA_CODE", text, re.DOTALL)
    package = 'package seu.deepseek.cx;'
    code_block = java_code_blocks[0].strip()
    code_block = package + '\n' + code_block

    # Determine the save directory
    global i
    save_dir = f"proxy_examples/{i}"
    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"DesignPatternExample_{i}.java"
    i = i + 1
    save_path = os.path.join(save_dir, file_name)

    # Save the code to a file
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(code_block)

    print(f"Saved Java design pattern example to file: {save_path}")

for key in proxy_mode_keywords:
    messages = [{"role": "system", "content": "You are a helpful assistant"},
                {"role": "user",
                 "content": f"Generate an example of Proxy design pattern based on the keyword {key}, also please add the start marker START_JAVA_CODE and end marker END_JAVA_CODE in the code section"}]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    print(f"Completed generating example for keyword {key}")
    # Extract and save the Nth design pattern example
    extract_and_save_java_code(response)