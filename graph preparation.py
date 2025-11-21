import json
import os
import re
from gensim.models import Word2Vec
import numpy as np

# New method for the laboratory slicing tool
def read_and_preprocess_cpg_file(cpg_file_path):
    with open(cpg_file_path, 'r', encoding='utf-8') as jf:
        cpg_string = jf.read()
        # If needed, regular expression replacement can be added here
        data = json.loads(cpg_string)
    return data

def extract_features_from_cpg_data(data):
    sentences = []
    nodes = data['nodes']

    for node in nodes:
        node_label = node.get('labels', '').strip()
        properties = node.get('properties', {})
        node_texts = [node_label]
        for prop_key in ['Name', 'Code']:  # Adjust property keys according to the new JSON structure
            prop_value = properties.get(prop_key)
            if prop_value and isinstance(prop_value, str) and prop_value.strip() != '<empty>':
                node_texts.append(prop_value.strip())

        sentences.append(node_texts)

    return sentences

def train_word2vec(sentences):
    if sentences:
        word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)
        return word2vec_model
    else:
        return None

def generate_node_features(word2vec_model, nodes):
    node_features = {}

    for node in nodes:
        node_id = node['id']
        feature_vector = np.zeros(100)

        if 'labels' in node and node['labels'] in word2vec_model.wv:
            feature_vector += word2vec_model.wv[node['labels']]
        # for prop_key in ['Name', 'Code']:  # Adjust property keys according to the new JSON structure
        #     prop_value = node.get('properties', {}).get(prop_key)
        #     if isinstance(prop_value, str) and prop_value.strip() != '<empty>':
        #         if prop_value.strip() in word2vec_model.wv:
        #             feature_vector += word2vec_model.wv[prop_value.strip()]

        feature_vector = feature_vector / np.linalg.norm(feature_vector) if np.linalg.norm(feature_vector) > 0 else feature_vector
        node_features[node_id] = feature_vector.tolist()  # Convert to list for JSON serialization

    return node_features

def process_cpg_files(root_dir):
    all_java_files_data = []

    # Collect data from all CPG files for training
    all_sentences = []
    for design_pattern in os.listdir(root_dir):
        design_pattern_path = os.path.join(root_dir, design_pattern)
        if os.path.isdir(design_pattern_path):
            for java_file in os.listdir(design_pattern_path):
                java_file_path = os.path.join(design_pattern_path, java_file)
                if os.path.isdir(java_file_path):
                    for cpg_file in os.listdir(java_file_path):
                        cpg_file_path = os.path.join(java_file_path, cpg_file)
                        if cpg_file.endswith('.json'):
                            data = read_and_preprocess_cpg_file(cpg_file_path)
                            sentences = extract_features_from_cpg_data(data)
                            all_sentences.extend(sentences)

    # Train Word2Vec model
    word2vec_model = train_word2vec(all_sentences)

    # If model training fails, return an empty list
    if not word2vec_model:
        return all_java_files_data

    # Iterate through all CPG files again to generate feature vectors and build JSON data
    for design_pattern in os.listdir(root_dir):
        design_pattern_path = os.path.join(root_dir, design_pattern)
        if os.path.isdir(design_pattern_path):
            for java_file in os.listdir(design_pattern_path):
                java_file_data = {
                    'java_file_name': java_file,
                    'node_features': [],
                    'graph': [],
                    'target': design_pattern
                }
                java_file_path = os.path.join(design_pattern_path, java_file)
                if os.path.isdir(java_file_path):
                    for cpg_file in os.listdir(java_file_path):
                        cpg_file_path = os.path.join(java_file_path, cpg_file)
                        if cpg_file.endswith('.json'):
                            data = read_and_preprocess_cpg_file(cpg_file_path)
                            nodes = data['nodes']
                            edges = data['edges']
                            node_features = generate_node_features(word2vec_model, nodes)
                            java_file_data['node_features'].extend(node_features.items())
                            java_file_data['graph'].extend(
                                [(e['source'], e['type'], e['target']) for e in edges])
                    all_java_files_data.append(java_file_data)

    return all_java_files_data

# Assume your CPG files are stored in a directory (abstracted path)
root_dir = "[Your Input Directory]"  # Abstracted input directory
aggregated_data = process_cpg_files(root_dir)

# Write the aggregated data to a large JSON file (abstracted path)
with open('[Your Output File Path]', 'w', encoding='utf-8') as f:  # Abstracted output file path
    json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

print("Processing completed. Results have been saved to the specified output file.")