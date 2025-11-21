import json
from time import localtime

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import dgl
from collections import defaultdict
from dgl import batch
import time
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Read JSON data
utc_now1 = time.localtime()


def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Create DGL graph and process node features and edge types
def create_dgl_graph_with_mapped_ids(node_features, edges):
    all_features = [features for _, features in node_features]
    node_id_to_idx = {node_id: idx for idx, (node_id, _) in enumerate(node_features)}
    num_nodes = len(node_id_to_idx)

    g = dgl.graph(([], []))
    g = dgl.add_self_loop(g)
    g.add_nodes(num_nodes)
    g.ndata['feat'] = torch.tensor(all_features, dtype=torch.float32)

    # Update edge indices and add edge type one-hot encoding
    updated_edges = [(node_id_to_idx[edge[0]], edge[1], node_id_to_idx[edge[2]]) for edge in edges]

    for src in updated_edges:
        g.add_edges(src[0], src[2])
        # Get one-hot encoding based on edge type
        etype_idx = edge_type_to_idx[src[1]]

    edge_type_one_hots = [edge_type_to_idx[src[1]] for src in updated_edges]
    # Convert the collected one-hot encoding list to a tensor
    g.edata['type'] = torch.tensor(edge_type_one_hots)
    return g


def process_data(data):
    graphs = []
    labels = []
    for index, item in enumerate(data):
        utc_now = time.localtime()
        print(
            f"Progress: {index + 1}/{len(data)}   now time is {utc_now}   node number is {len(item['node_features'])}  edge number is {len(item['graph'])}")
        g = create_dgl_graph_with_mapped_ids(item['node_features'], item['graph'])
        graphs.append(g)
        labels.append(class_map[item['target']])

    return graphs, labels


# Data splitting
def split_data(graphs, labels, train_size, random_state=41):
    # Use StratifiedShuffleSplit for stratified sampling
    # n_splits=1 means only one split, train_size indicates the proportion of the training set in the total dataset
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)

    for train_idx, val_idx in sss.split(np.zeros(len(labels)), labels):
        pass  # Only take the first split

    # Split data using indices
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    print("Original data class distribution:", Counter(labels))
    print("Training set class distribution:", Counter(train_labels))
    print("Validation set class distribution:", Counter(val_labels))

    return train_graphs, val_graphs, train_labels, val_labels


# Create DGL dataset
class DGLDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        label = self.labels[idx]
        return g, torch.tensor(label, dtype=torch.long)


def collate_fn(batch_data):
    # batch_data is a list containing several (g, labels) pairs
    graph_list, label_list = zip(*batch_data)  # Unzip graphs and labels
    # Use DGL's batch function to merge multiple graphs into one batch graph
    batched_graph = batch(graph_list)
    # Convert label list to tensor
    labels = torch.tensor(label_list, dtype=torch.long)
    return batched_graph, labels


class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_etypes, n_steps, dropout_rate):
        super().__init__()
        self.conv1 = dgl.nn.GATv2Conv(
            hidden_dim,
            hidden_dim,
            num_heads=4,
            feat_drop=dropout_rate,
            allow_zero_in_degree=True
        )
        self.dim_reducer = nn.Linear(hidden_dim * 4,
                                     hidden_dim)  # GatedGraphConv layer - use original edge type indices
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = dgl.nn.GatedGraphConv(
            input_dim,
            hidden_dim,
            n_steps=n_steps,
            n_etypes=n_etypes  # Use original 20 types
        )

        # Skip connection
        self.skip_lin = nn.Linear(input_dim, hidden_dim * 4) if input_dim != hidden_dim * 4 else nn.Identity()

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Attention pooling
        self.attn_pool = dgl.nn.GlobalAttentionPooling(nn.Linear(hidden_dim, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, g):
        # Node features
        h = g.ndata['feat']
        h2 = self.conv2(g, h, g.edata['type'])  # Directly pass integer indices

        h2 = F.elu(h2)
        h2 = self.bn2(h2)
        # First convolution - GAT
        h1 = self.conv1(g, h2)  # Output shape: [num_nodes, num_heads, hidden_dim]
        h1 = F.elu(h1)

        # Merge multiple heads - Concatenate multiple head outputs
        h1 = h1.view(-1, h1.size(1) * h1.size(2))  # Shape becomes: [num_nodes, num_heads * hidden_dim]

        # Skip connection
        skip = self.skip_lin(h)  # Convert original features to the same dimension
        h1 = h1 + skip

        # Batch normalization
        h1 = self.bn1(h1)

        # Second convolution - GatedGraphConv
        # Directly use original edge type indices

        h1_reduced = self.dim_reducer(h1)

        # Attention pooling
        with g.local_scope():
            g.ndata['h'] = h1_reduced
            h_g = self.attn_pool(g, g.ndata['h'])

        # Classification
        return self.classifier(h_g)


class onlyGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_etypes, n_steps, dropout_rate):
        super().__init__()
        self.conv1 = dgl.nn.GATv2Conv(
            input_dim,
            hidden_dim,
            num_heads=2,
            feat_drop=dropout_rate,
            allow_zero_in_degree=True
        )
        self.dim_reducer = nn.Linear(hidden_dim * 2,
                                     hidden_dim)  # GatedGraphConv layer - use original edge type indices
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = dgl.nn.GatedGraphConv(
            hidden_dim,
            hidden_dim,
            n_steps=n_steps,
            n_etypes=n_etypes  # Use original 20 types
        )

        # Skip connection
        self.skip_lin = nn.Linear(input_dim, hidden_dim * 2) if input_dim != hidden_dim * 2 else nn.Identity()

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Attention pooling
        self.attn_pool = dgl.nn.GlobalAttentionPooling(nn.Linear(hidden_dim, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, g):
        # Node features
        h = g.ndata['feat']

        # First convolution - GAT
        h1 = self.conv1(g, h)  # Output shape: [num_nodes, num_heads, hidden_dim]
        h1 = F.elu(h1)

        # Merge multiple heads - Concatenate multiple head outputs
        h1 = h1.view(-1, h1.size(1) * h1.size(2))  # Shape becomes: [num_nodes, num_heads * hidden_dim]

        # Skip connection
        skip = self.skip_lin(h)  # Convert original features to the same dimension
        h1 = h1 + skip

        # Batch normalization
        h1 = self.bn1(h1)

        h1_reduced = self.dim_reducer(h1)

        # Attention pooling
        with g.local_scope():
            g.ndata['h'] = h1_reduced
            h_g = self.attn_pool(g, g.ndata['h'])

        # Classification
        return self.classifier(h_g)


class onlyGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_etypes, n_steps, dropout_rate):
        super().__init__()
        self.conv1 = dgl.nn.GATv2Conv(
            input_dim,
            hidden_dim,
            num_heads=4,
            feat_drop=dropout_rate,
            allow_zero_in_degree=True
        )
        self.dim_reducer = nn.Linear(hidden_dim * 4,
                                     hidden_dim)  # GatedGraphConv layer - use original edge type indices
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = dgl.nn.GatedGraphConv(
            hidden_dim,
            hidden_dim,
            n_steps=n_steps,
            n_etypes=n_etypes  # Use original 20 types
        )

        # Skip connection
        self.skip_lin = nn.Linear(input_dim, hidden_dim * 4) if input_dim != hidden_dim * 4 else nn.Identity()

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Attention pooling
        self.attn_pool = dgl.nn.GlobalAttentionPooling(nn.Linear(hidden_dim, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, g):
        # Node features
        h = g.ndata['feat']

        h2 = self.conv2(g, h, g.edata['type'])  # Directly pass integer indices

        h2 = F.elu(h2)
        h2 = self.bn2(h2)

        # Attention pooling
        with g.local_scope():
            g.ndata['h'] = h2
            h_g = self.attn_pool(g, g.ndata['h'])

        # Classification
        return self.classifier(h_g)


def custom_classification_report(true_labels, pred_labels, class_names):
    # Ensure class names are in the correct order
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    true_encoded = label_encoder.transform(true_labels)
    pred_encoded = label_encoder.transform(pred_labels)

    # Calculate various metrics
    accuracy = accuracy_score(true_encoded, pred_encoded)
    precision = precision_score(true_encoded, pred_encoded, average=None)
    recall = recall_score(true_encoded, pred_encoded, average=None)
    f1 = f1_score(true_encoded, pred_encoded, average=None)

    # Generate report
    report = {
        "accuracy": accuracy,
        "classes": class_names,
        "precision": {class_name: precision[i] for i, class_name in enumerate(class_names)},
        "recall": {class_name: recall[i] for i, class_name in enumerate(class_names)},
        "f1-score": {class_name: f1[i] for i, class_name in enumerate(class_names)}
    }

    # Print report
    print(f"Accuracy: {accuracy:.4f}")
    print("{:<15} {:<15} {:<15} {:<15}".format("Class", "Precision", "Recall", "F1-Score"))
    for class_name in class_names:
        print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            class_name,
            report["precision"][class_name],
            report["recall"][class_name],
            report["f1-score"][class_name]
        ))

    return report


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, loader, class_names, epoch, hdim, dropout, lr, st):
    model.eval()
    true_labels = []
    pred_labels = []
    a_time = time.time()  # Record end time of training

    with torch.no_grad():
        for g, labels in loader:
            g = g.to(device)
            labels = labels.to(device)
            logits = model(g)
            _, predicted = torch.max(logits.data, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    # Convert numerical labels to string labels
    true_labels_str = [class_names[label] for label in true_labels]
    pred_labels_str = [class_names[label] for label in pred_labels]

    # 1. Generate classification report
    report = custom_classification_report(true_labels_str, pred_labels_str, class_names)
    b_time = time.time()
    c_time = b_time - a_time
    print(f'Training took {c_time:.2f} seconds')  # 2. Generate confusion matrix
    cm = confusion_matrix(true_labels_str, pred_labels_str, labels=class_names)

    # 3. Save results to file
    filename = f"SDG-GNN\\result\\{utc_now1}type1r6_report.txt"
    with open(filename, "a") as f:
        # Write hyperparameter information
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Epoch: {epoch + 1} | Hidden Dim: {hdim} | Dropout: {dropout} | LR: {lr} | Steps: {st}\n")
        f.write(f"{'=' * 80}\n\n")

        # Write classification report
        f.write("Classification Report:\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        f.write("{:<15} {:<15} {:<15} {:<15}\n".format("Class", "Precision", "Recall", "F1-Score"))
        for class_name in class_names:
            f.write("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}\n".format(
                class_name,
                report["precision"][class_name],
                report["recall"][class_name],
                report["f1-score"][class_name]
            ))
        f.write(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))

        # Write confusion matrix
        f.write("\nConfusion Matrix:\n")
        f.write("\t" + "\t".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]}\t" + "\t".join(map(str, row)) + "\n")

        # Write matrix visualization
        f.write("\nConfusion Matrix Visualization:\n")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f"SDG-GNN\\result\\{utc_now1}cm_epoch{epoch + 1}.png")
        plt.close()
        f.write(f"Confusion matrix saved as cm_epoch{epoch + 1}.png\n\n")

    # Return report and matrix
    return report, cm


if __name__ == "__main__":

    start_time = time.time()
    torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Start reading data')

    json_file = 'SDG-GNN//318json.json'  # Replace with your JSON file path
    data = load_data(json_file)
    all_edge_types = ['pkgDepd',
                      'pkgMember',
                      'classMember',
                      'dataMember',
                      'controlDepd',
                      'parameter',
                      'toObject',
                      'dataDepd',
                      'typeAnalysis',
                      'objMember',
                      'initDataDependenceEdge',
                      'methodInvocation',
                      'inputParameter',
                      'outputParameter',
                      'summaryEdge',
                      'abstractMember',
                      'interfaceImplement',
                      'methodImplment',
                      'inherit',
                      'EnumConstant',
                      'SELF_LOOP']

    dis_label = ['FactoryMethod', 'Singleton', 'Adapter',
                 'AbstractFactory', 'Decorator', 'Facade', 'Proxy']

    edge_type_to_idx = {etype: idx for idx, etype in enumerate(all_edge_types)}
    n_etypes = len(all_edge_types)

    print('Start processing data')
    class_map = {cls_name: idx for idx, cls_name in enumerate(set(dis_label))}

    graphs, labels = process_data(data)

    print('Split dataset')
    train_graphs, graphs, train_labels, labels = split_data(graphs, labels, 0.7)
    test_graphs, val_graphs, test_labels, val_labels = split_data(graphs, labels, 0.5)

    # Create data loaders
    print('Create data loaders')
    train_dataset = DGLDataset(train_graphs, train_labels)
    val_dataset = DGLDataset(val_graphs, val_labels)
    test_dataset = DGLDataset(test_graphs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    # Model parameters
    input_dim = 100  # Dimension of node features
    hidden_dim = 128  # Dimension of hidden layer
    num_classes = 7  # Number of classes
    # Without adding method hierarchy dependency edges
    # n_etypes = 20
    print('Create data loaders')
    hidden_dims = [128]
    dropouts = [0.4]
    learning_rates = [0.001, 0.0013, 0.0015]
    steps = [2]
    for i in range(1, 2):
        for h_dim in hidden_dims:
            for dropout in dropouts:
                for lr in learning_rates:
                    for st in steps:

                        model = GNNClassifier(input_dim, h_dim, num_classes, n_etypes, st, dropout)
                        total_params = sum(p.numel() for p in model.parameters())
                        print("Total Parameters:", total_params)
                        model = model.to(device)
                        utc_now1 = time.localtime()
                        # Define loss function and optimizer
                        cross_entropy = nn.CrossEntropyLoss()
                        optimizer = Adam(model.parameters(), lr=lr)
                        optimizer = optimizer


                        # Train model
                        def train_epoch(model, loader, optimizer, loss_fn):
                            model.train()
                            total_loss = 0
                            for step, (g, labels) in enumerate(loader):
                                # g = augment_graph(g)
                                g = g.to(device)
                                labels = labels.to(device)
                                optimizer.zero_grad()
                                logits = model(g)
                                loss = loss_fn(logits, labels)
                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item()
                            return total_loss / len(loader)


                        # Evaluate model
                        def eval_epoch(model, loader, loss_fn):
                            model.eval()
                            total_loss = 0
                            with torch.no_grad():
                                for g, labels in loader:
                                    g = g.to(device)
                                    labels = labels.to(device)
                                    logits = model(g)
                                    loss = loss_fn(logits, labels)
                                    total_loss += loss.item()
                            return total_loss / len(loader)


                        class_names = dis_label  # Ensure this list contains all class names
                        test_results = []
                        num_epochs = 500  # Number of training epochs
                        for epoch in range(num_epochs):
                            train_loss = train_epoch(model, train_loader, optimizer, cross_entropy)
                            val_loss = eval_epoch(model, val_loader, cross_entropy)

                            print(f'Epoch {epoch + 1:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                            if (epoch + 1) % 15 == 0:
                                test_accuracy_report = evaluate_model(model, test_loader, class_names, epoch, h_dim,
                                                                      dropout, lr, st)
                                test_results.append(test_accuracy_report)

    end_time = time.time()  # Record end time of training
    train_time = end_time - start_time  # Calculate time spent on training
    print(f'Training took {train_time:.2f} seconds')