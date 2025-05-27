import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
from tqdm import tqdm
import pandas as pd

# Tokenizer
def simple_tokenizer(text):
    return text.strip().split()

def extract_floats(text):
    return [float(tok) for tok in text.strip().split()]

# Dataset
class AvgDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = simple_tokenizer(sample['input'])
        targets = extract_floats(sample['output'])
        return tokens, torch.tensor(targets)

def collate_fn(batch):
    tokens, targets = zip(*batch)
    return list(tokens), torch.stack(targets)

# Embedding
class FloatEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, tokens, device):
        batch_embeddings = []
        for seq in tokens:
            seq_embeddings = []
            for token in seq:
                vec = torch.zeros(self.embedding_dim, device=device)
                try:
                    vec[0] = float(token)
                except:
                    vec[0] = -999.0
                seq_embeddings.append(vec)
            batch_embeddings.append(torch.stack(seq_embeddings))
        return torch.stack(batch_embeddings)

# Model
class GPT2AvgPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = FloatEmbedding(config.n_embd)
        self.transformer = GPT2Model(config)

    def forward(self, input_tokens, device):
        embeds = self.embedding(input_tokens, device)
        attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        output = self.transformer(inputs_embeds=embeds, attention_mask=attention_mask)
        return output.last_hidden_state

# Evaluation: only compare squared mean prediction
@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    count = 0
    for tokens, gt in dataloader:
        pred_mean = model(tokens, device)[:, -1, 0]
        tokens_with_mean = [seq + [f"{pred_mean[i].item():.3f}"] for i, seq in enumerate(tokens)]
        pred_sq = model(tokens_with_mean, device)[:, -1, 0]
        loss = loss_fn(pred_sq.to(device), gt[:, 1].to(device))
        total_loss += loss.item()
        count += 1
    return total_loss / count

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPT2Config(n_embd=64, n_layer=8, n_head=4, vocab_size=1)
    model = GPT2AvgPredictor(config).to(device)
    model.load_state_dict(torch.load("best_model_mean_squared.pt", map_location=device))

    loss_fn = nn.MSELoss()
    results = {}

    for i in range(6):
        path = f"ood_testing_data_shift{i}.jsonl"
        dataset = AvgDataset(path)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        mse = evaluate(model, loader, device, loss_fn)
        results[f"shift_{i}"] = mse
        print(f"shift_{i}: {mse:.6f}")

    df = pd.DataFrame(list(results.items()), columns=["Dataset", "MSE"])
    df.to_csv("eval_mean_squared_results.csv", index=False)
    print("\nSaved to eval_mean_squared_results.csv")

if __name__ == '__main__':
    main()
