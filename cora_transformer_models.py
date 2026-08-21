# -*- coding: utf-8 -*-
"""
Cora graph sequence experiments.

Experiments:
1. Past labels -> next label.
2. Past node features -> future node features.
3. Past node features -> future node labels.
4. Strict node-split future-label appendix using Cora train/val/test masks.

The main experiments use a random-walk split. The appendix uses walks restricted
entirely to one official Cora mask, so nodes cannot cross train/validation/test
splits.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx


# =============================================================================
# Configuration and reproducibility
# =============================================================================
SEED = 42
WALK_LENGTH = 8
PAST_LEN = 4
FUTURE_LEN = 4
BATCH_SIZE = 32
LABEL_BATCH_SIZE = 64
LABEL_EPOCHS = 10
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data
# =============================================================================
def load_cora():
    dataset = Planetoid(root="./data/Cora", name="Cora")
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True)
    features = data.x.cpu().numpy().astype(np.float32)
    labels = data.y.cpu().numpy().astype(np.int64)
    class_ids = list(range(dataset.num_classes))
    class_names = [f"class_{i}" for i in class_ids]

    print("Device:", device)
    print(f"Dataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges in PyG edge_index: {data.num_edges}")
    print(f"Features: {dataset.num_features}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")

    return dataset, data, graph, features, labels, class_ids, class_names


def generate_node_walks(graph, num_walks=5, walk_length=8):
    walks = []
    nodes = list(graph.nodes())

    for _ in range(num_walks):
        random.shuffle(nodes)
        for start_node in nodes:
            walk = [int(start_node)]
            current = int(start_node)

            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(current))
                if not neighbors:
                    break
                current = int(random.choice(neighbors))
                walk.append(current)

            if len(walk) == walk_length:
                walks.append(walk)

    return walks


def split_walks(walks):
    train_walks, temporary_walks = train_test_split(
        walks, test_size=0.30, random_state=SEED
    )
    val_walks, test_walks = train_test_split(
        temporary_walks, test_size=0.50, random_state=SEED
    )
    return train_walks, val_walks, test_walks


# =============================================================================
# Experiment 1: past labels -> next label
# =============================================================================
class LabelNextDataset(Dataset):
    def __init__(self, walks, labels, num_classes, sequence_length):
        self.examples = []
        eye = torch.eye(num_classes)

        for walk in walks:
            label_walk = [int(labels[node]) for node in walk]
            for start in range(len(label_walk) - sequence_length):
                source = label_walk[start:start + sequence_length]
                target = label_walk[start + sequence_length]
                self.examples.append(
                    (
                        eye[torch.tensor(source)].float(),
                        torch.tensor(target, dtype=torch.long),
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        x, y = self.examples[index]
        return {"input_features": x, "labels": y}


class BertNextLabelEncoder(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=2, num_heads=4):
        super().__init__()
        self.projection = nn.Linear(num_classes, hidden_size)
        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.bert = BertModel(config)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_features, labels=None):
        projected = self.projection(input_features)
        mask = torch.ones(
            projected.shape[:2], dtype=torch.long, device=projected.device
        )
        encoded = self.bert(
            inputs_embeds=projected, attention_mask=mask
        ).last_hidden_state
        logits = self.classifier(encoded[:, 0, :])
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}


@torch.no_grad()
def evaluate_next_label(model, loader, class_ids, class_names):
    model.eval()
    true_values, predicted_values = [], []

    for batch in loader:
        x = batch["input_features"].to(device)
        y = batch["labels"].to(device)
        prediction = model(x)["logits"].argmax(dim=-1)
        true_values.extend(y.cpu().numpy())
        predicted_values.extend(prediction.cpu().numpy())

    accuracy = float(np.mean(np.asarray(true_values) == np.asarray(predicted_values)))
    print("\nExperiment 1: Next-label evaluation")
    print("Accuracy:", round(accuracy, 4))
    print(
        "Confusion matrix:\n",
        confusion_matrix(true_values, predicted_values, labels=class_ids),
    )
    print(
        classification_report(
            true_values,
            predicted_values,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )
    )
    return accuracy


def run_experiment_1(train_walks, val_walks, test_walks, labels, num_classes, class_ids, class_names):
    train_dataset = LabelNextDataset(train_walks, labels, num_classes, PAST_LEN)
    val_dataset = LabelNextDataset(val_walks, labels, num_classes, PAST_LEN)
    test_dataset = LabelNextDataset(test_walks, labels, num_classes, PAST_LEN)

    train_loader = DataLoader(train_dataset, batch_size=LABEL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=LABEL_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=LABEL_BATCH_SIZE)

    print(
        f"\nExperiment 1 dataset | train {len(train_dataset)} | "
        f"val {len(val_dataset)} | test {len(test_dataset)}"
    )

    model = BertNextLabelEncoder(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, LABEL_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["input_features"].to(device)
            y = batch["labels"].to(device)
            loss = model(x, y)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_label_loss(model, val_loader)
        print(
            f"Experiment 1 epoch {epoch:02d} | "
            f"train loss {total_loss / max(1, len(train_loader)):.4f} | "
            f"val loss {val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    evaluate_next_label(model, test_loader, class_ids, class_names)


def evaluate_label_loss(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["input_features"].to(device)
            y = batch["labels"].to(device)
            total_loss += model(x, y)["loss"].item()
    return total_loss / max(1, len(loader))


# =============================================================================
# Shared future-walk dataset
# =============================================================================
class FutureWalkDataset(Dataset):
    """Return x1...x4, x5...x8, and y5...y8 for each walk."""

    def __init__(self, walks, features, labels, past_len, future_len):
        self.examples = []
        required_length = past_len + future_len

        for walk in walks:
            if len(walk) < required_length:
                continue
            past_nodes = walk[:past_len]
            future_nodes = walk[past_len:required_length]
            self.examples.append(
                {
                    "past_features": torch.tensor(
                        features[past_nodes], dtype=torch.float32
                    ),
                    "future_features": torch.tensor(
                        features[future_nodes], dtype=torch.float32
                    ),
                    "future_labels": torch.tensor(
                        labels[future_nodes], dtype=torch.long
                    ),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# =============================================================================
# Experiment 2: past features -> future features
# =============================================================================
class FutureFeatureSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, nhead=4,
                 past_len=4, future_len=4, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.src_proj = nn.Linear(input_dim, hidden_dim)
        self.tgt_proj = nn.Linear(input_dim, hidden_dim)
        self.src_pos = nn.Embedding(past_len, hidden_dim)
        self.tgt_pos = nn.Embedding(future_len, hidden_dim)
        config = BertConfig(
            hidden_size=hidden_dim,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        self.encoder = BertModel(config)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def causal_mask(self, length, target_device):
        return nn.Transformer.generate_square_subsequent_mask(length).to(target_device)

    def encode(self, past_features):
        batch_size, sequence_length, _ = past_features.shape
        position_ids = torch.arange(sequence_length, device=past_features.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        source = self.src_proj(past_features) + self.src_pos(position_ids)
        source = self.dropout(source)
        mask = torch.ones(
            batch_size, sequence_length, dtype=torch.long, device=past_features.device
        )
        return self.encoder(inputs_embeds=source, attention_mask=mask).last_hidden_state

    def forward(self, past_features, decoder_input_features):
        batch_size, target_length, _ = decoder_input_features.shape
        memory = self.encode(past_features)
        position_ids = torch.arange(target_length, device=past_features.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        target = self.tgt_proj(decoder_input_features) + self.tgt_pos(position_ids)
        target = self.dropout(target)
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=self.causal_mask(target_length, past_features.device),
        )
        return self.output(decoded)

    @torch.no_grad()
    def generate(self, past_features, future_len):
        batch_size = past_features.size(0)
        generated = torch.zeros(
            batch_size, 1, self.input_dim, device=past_features.device
        )
        for _ in range(future_len):
            prediction = self.forward(past_features, generated)
            generated = torch.cat([generated, prediction[:, -1:, :]], dim=1)
        return generated[:, 1:, :]


def make_feature_decoder_input(future_features):
    zeros = torch.zeros(
        future_features.size(0), 1, future_features.size(2),
        device=future_features.device,
    )
    return torch.cat([zeros, future_features[:, :-1, :]], dim=1)


def cosine_similarity_batch(prediction, target):
    return F.cosine_similarity(
        prediction.reshape(-1, prediction.size(-1)),
        target.reshape(-1, target.size(-1)),
        dim=-1,
    ).mean().item()


@torch.no_grad()
def nearest_neighbor_label_accuracy(prediction, true_labels, candidate_features, candidate_labels):
    predicted = F.normalize(prediction.reshape(-1, prediction.size(-1)), dim=-1)
    candidates = F.normalize(candidate_features, dim=-1)
    nearest = (predicted @ candidates.T).argmax(dim=-1)
    return (candidate_labels[nearest] == true_labels.reshape(-1)).float().mean().item()


@torch.no_grad()
def evaluate_future_features(model, loader, candidate_features, candidate_labels):
    model.eval()
    mse_values, cosine_values, label_values = [], [], []
    for batch in loader:
        past = batch["past_features"].to(device)
        future = batch["future_features"].to(device)
        labels = batch["future_labels"].to(device)
        prediction = model.generate(past, future.size(1))
        mse_values.append(F.mse_loss(prediction, future).item())
        cosine_values.append(cosine_similarity_batch(prediction, future))
        label_values.append(
            nearest_neighbor_label_accuracy(
                prediction, labels, candidate_features, candidate_labels
            )
        )
    return (
        float(np.mean(mse_values)) if mse_values else float("inf"),
        float(np.mean(cosine_values)) if cosine_values else 0.0,
        float(np.mean(label_values)) if label_values else 0.0,
    )


def run_experiment_2(train_loader, val_loader, test_loader, features, labels):
    print("\nExperiment 2: Past features -> future features")
    model = FutureFeatureSeq2Seq(
        features.shape[1], past_len=PAST_LEN, future_len=FUTURE_LEN
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    candidate_features = torch.tensor(features, dtype=torch.float32, device=device)
    candidate_labels = torch.tensor(labels, dtype=torch.long, device=device)
    best_state = None
    best_val_mse = float("inf")
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            past = batch["past_features"].to(device)
            future = batch["future_features"].to(device)
            prediction = model(past, make_feature_decoder_input(future))
            loss = F.mse_loss(prediction, future)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_mse, val_cosine, val_nn = evaluate_future_features(
            model, val_loader, candidate_features, candidate_labels
        )
        print(
            f"Experiment 2 epoch {epoch:02d} | train MSE {np.mean(losses):.6f} | "
            f"val MSE {val_mse:.6f} | val cosine {val_cosine:.4f} | "
            f"val NN-label acc {val_nn:.4f}"
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Experiment 2 early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_future_features(
        model, test_loader, candidate_features, candidate_labels
    )
    print("\nExperiment 2 TEST")
    print(f"MSE: {test_metrics[0]:.6f}")
    print(f"Cosine similarity: {test_metrics[1]:.4f}")
    print(f"Nearest-neighbor label accuracy: {test_metrics[2]:.4f}")


# =============================================================================
# Experiment 3: past features -> future labels
# =============================================================================
class FutureLabelSeq2Seq(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2,
                 nhead=4, past_len=4, future_len=4, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.bos_idx = num_classes
        self.src_proj = nn.Linear(input_dim, hidden_dim)
        self.src_pos = nn.Embedding(past_len, hidden_dim)
        config = BertConfig(
            hidden_size=hidden_dim,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        self.encoder = BertModel(config)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.token_embed = nn.Embedding(num_classes + 1, hidden_dim)
        self.tgt_pos = nn.Embedding(future_len, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def causal_mask(self, length, target_device):
        return nn.Transformer.generate_square_subsequent_mask(length).to(target_device)

    def encode(self, past_features):
        batch_size, sequence_length, _ = past_features.shape
        positions = torch.arange(sequence_length, device=past_features.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        source = self.src_proj(past_features) + self.src_pos(positions)
        source = self.dropout(source)
        mask = torch.ones(
            batch_size, sequence_length, dtype=torch.long, device=past_features.device
        )
        return self.encoder(inputs_embeds=source, attention_mask=mask).last_hidden_state

    def forward(self, past_features, decoder_input_labels):
        batch_size, target_length = decoder_input_labels.shape
        memory = self.encode(past_features)
        positions = torch.arange(target_length, device=past_features.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        target = self.token_embed(decoder_input_labels) + self.tgt_pos(positions)
        target = self.dropout(target)
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=self.causal_mask(target_length, past_features.device),
        )
        return self.classifier(decoded)

    @torch.no_grad()
    def generate(self, past_features, future_len):
        sequence = torch.full(
            (past_features.size(0), 1), self.bos_idx,
            dtype=torch.long, device=past_features.device,
        )
        for _ in range(future_len):
            logits = self.forward(past_features, sequence)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sequence = torch.cat([sequence, next_token], dim=1)
        return sequence[:, 1:]


def make_label_decoder_input(labels, bos_idx):
    bos = torch.full(
        (labels.size(0), 1), bos_idx,
        dtype=torch.long, device=labels.device,
    )
    return torch.cat([bos, labels[:, :-1]], dim=1)


@torch.no_grad()
def evaluate_future_labels(model, loader, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0
    true_values, predicted_values = [], []

    for batch in loader:
        past = batch["past_features"].to(device)
        labels = batch["future_labels"].to(device)
        decoder_input = make_label_decoder_input(labels, model.bos_idx)
        logits = model(past, decoder_input)
        prediction = logits.argmax(dim=-1)
        total_loss += criterion(logits.reshape(-1, num_classes), labels.reshape(-1)).item()
        total_tokens += labels.numel()
        correct_tokens += (prediction == labels).sum().item()
        total_sequences += labels.size(0)
        correct_sequences += (prediction == labels).all(dim=1).sum().item()
        true_values.extend(labels.cpu().numpy().reshape(-1))
        predicted_values.extend(prediction.cpu().numpy().reshape(-1))

    return (
        total_loss / max(1, len(loader)),
        correct_tokens / max(1, total_tokens),
        correct_sequences / max(1, total_sequences),
        np.asarray(true_values),
        np.asarray(predicted_values),
    )


@torch.no_grad()
def evaluate_generated_labels(model, loader, class_ids, class_names, title):
    model.eval()
    true_sequences, predicted_sequences = [], []
    shown = 0
    for batch in loader:
        past = batch["past_features"].to(device)
        labels = batch["future_labels"].to(device)
        prediction = model.generate(past, labels.size(1))
        true_sequences.extend(labels.cpu().numpy())
        predicted_sequences.extend(prediction.cpu().numpy())
        while shown < 5 and shown < labels.size(0):
            print(f"\n{title} sample {shown + 1}")
            print("True future labels:", labels[shown].cpu().numpy())
            print("Pred future labels:", prediction[shown].cpu().numpy())
            shown += 1

    true_array = np.asarray(true_sequences)
    predicted_array = np.asarray(predicted_sequences)
    token_accuracy = float(np.mean(true_array == predicted_array))
    sequence_accuracy = float(np.mean(np.all(true_array == predicted_array, axis=1)))
    print(f"\n{title} generated evaluation")
    print("Generated token accuracy:", token_accuracy)
    print("Generated sequence accuracy:", sequence_accuracy)
    print(
        classification_report(
            true_array.reshape(-1), predicted_array.reshape(-1),
            labels=class_ids, target_names=class_names, zero_division=0,
        )
    )
    return token_accuracy, sequence_accuracy


def train_future_label_model(train_loader, val_loader, test_loader, num_classes,
                             class_ids, class_names, title="Experiment 3"):
    print(f"\n{title}: Past features -> future labels")
    model = FutureLabelSeq2Seq(
        feat_dim, num_classes, past_len=PAST_LEN, future_len=FUTURE_LEN
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            past = batch["past_features"].to(device)
            labels = batch["future_labels"].to(device)
            decoder_input = make_label_decoder_input(labels, model.bos_idx)
            logits = model(past, decoder_input)
            loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_loss, val_token, val_sequence, _, _ = evaluate_future_labels(
            model, val_loader, num_classes
        )
        print(
            f"{title} epoch {epoch:02d} | train loss {np.mean(losses):.4f} | "
            f"val loss {val_loss:.4f} | val token acc {val_token:.4f} | "
            f"val seq acc {val_sequence:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"{title} early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    loss, token, sequence, true_values, predicted_values = evaluate_future_labels(
        model, test_loader, num_classes
    )
    print(f"\n{title} TEST")
    print(f"Loss: {loss:.4f}")
    print(f"Teacher-forced token accuracy: {token:.4f}")
    print(f"Teacher-forced sequence accuracy: {sequence:.4f}")
    print(
        classification_report(
            true_values, predicted_values,
            labels=class_ids, target_names=class_names, zero_division=0,
        )
    )
    evaluate_generated_labels(model, test_loader, class_ids, class_names, title)
    return model


# =============================================================================
# Strict node-split appendix
# =============================================================================
def generate_strict_node_split_walks(graph, allowed_nodes, num_walks, walk_length):
    allowed = {int(node) for node in allowed_nodes}
    starts = list(allowed)
    walks = []

    for _ in range(num_walks):
        random.shuffle(starts)
        for start in starts:
            walk = [start]
            current = start
            for _ in range(walk_length - 1):
                neighbors = [
                    int(node) for node in graph.neighbors(current)
                    if int(node) in allowed
                ]
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            if len(walk) == walk_length:
                walks.append(walk)
    return walks


def make_strict_splits(graph, data):
    train_nodes = torch.where(data.train_mask)[0].cpu().numpy()
    val_nodes = torch.where(data.val_mask)[0].cpu().numpy()
    test_nodes = torch.where(data.test_mask)[0].cpu().numpy()
    print(
        f"Strict split nodes | train {len(train_nodes)} | "
        f"val {len(val_nodes)} | test {len(test_nodes)}"
    )

    configurations = [
        (8, 4, 4, 300, "x1 x2 x3 x4 -> y5 y6 y7 y8"),
        (6, 3, 3, 400, "x1 x2 x3 -> y4 y5 y6"),
        (4, 2, 2, 600, "x1 x2 -> y3 y4"),
        (2, 1, 1, 1000, "x1 -> y2"),
    ]

    for walk_length, past_len, future_len, attempts, description in configurations:
        train = generate_strict_node_split_walks(graph, train_nodes, attempts, walk_length)
        val = generate_strict_node_split_walks(graph, val_nodes, attempts, walk_length)
        test = generate_strict_node_split_walks(graph, test_nodes, attempts, walk_length)
        print(
            f"Trying strict task {description} | train walks {len(train)} | "
            f"val walks {len(val)} | test walks {len(test)}"
        )
        if len(train) >= 50 and len(val) > 0 and len(test) > 0:
            return train, val, test, past_len, future_len, description

    return None


def run_strict_appendix(graph, data, features, labels, num_classes, class_ids, class_names):
    print("\n" + "=" * 80)
    print("APPENDIX: Strict node-split future-label prediction")
    print("=" * 80)
    result = make_strict_splits(graph, data)
    if result is None:
        print("Strict appendix skipped: no usable strict walks were generated.")
        return

    train, val, test, past_len, future_len, description = result
    train_dataset = FutureWalkDataset(train, features, labels, past_len, future_len)
    val_dataset = FutureWalkDataset(val, features, labels, past_len, future_len)
    test_dataset = FutureWalkDataset(test, features, labels, past_len, future_len)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print(
        f"Strict appendix task: {description} | "
        f"train {len(train_dataset)} | val {len(val_dataset)} | test {len(test_dataset)}"
    )

    global PAST_LEN, FUTURE_LEN
    old_past, old_future = PAST_LEN, FUTURE_LEN
    PAST_LEN, FUTURE_LEN = past_len, future_len
    try:
        train_future_label_model(
            train_loader, val_loader, test_loader, num_classes,
            class_ids, class_names, title="Strict appendix",
        )
    finally:
        PAST_LEN, FUTURE_LEN = old_past, old_future


# =============================================================================
# Main
# =============================================================================
def main():
    global feat_dim
    dataset, data, graph, features, labels, class_ids, class_names = load_cora()
    feat_dim = dataset.num_features

    all_walks = generate_node_walks(graph, num_walks=5, walk_length=WALK_LENGTH)
    train_walks, val_walks, test_walks = split_walks(all_walks)
    print(
        f"Node walks: {len(all_walks)} | train {len(train_walks)} | "
        f"val {len(val_walks)} | test {len(test_walks)}"
    )

    run_experiment_1(
        train_walks, val_walks, test_walks, labels,
        dataset.num_classes, class_ids, class_names,
    )

    train_dataset = FutureWalkDataset(
        train_walks, features, labels, PAST_LEN, FUTURE_LEN
    )
    val_dataset = FutureWalkDataset(
        val_walks, features, labels, PAST_LEN, FUTURE_LEN
    )
    test_dataset = FutureWalkDataset(
        test_walks, features, labels, PAST_LEN, FUTURE_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print(
        f"\nFuture prediction dataset | train {len(train_dataset)} | "
        f"val {len(val_dataset)} | test {len(test_dataset)}"
    )

    run_experiment_2(train_loader, val_loader, test_loader, features, labels)
    train_future_label_model(
        train_loader, val_loader, test_loader,
        dataset.num_classes, class_ids, class_names,
        title="Experiment 3",
    )
    run_strict_appendix(
        graph, data, features, labels,
        dataset.num_classes, class_ids, class_names,
    )


if __name__ == "__main__":
    main()
