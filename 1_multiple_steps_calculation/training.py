import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model, get_cosine_schedule_with_warmup
from tqdm import tqdm
import random

random.seed(42)
torch.manual_seed(42)

# Tokenizer (split by space)
def simple_tokenizer(text):
    return text.strip().split()

def extract_floats(text):
    return [float(tok) for tok in text.strip().split()]

# Custom float embedding
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

# GPT-2 based model
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
        targets = extract_floats(sample['output'])  # [mean, square]
        return tokens, torch.tensor(targets)

def collate_fn(batch):
    tokens, targets = zip(*batch)
    return list(tokens), torch.stack(targets)

# Evaluation (autoregressive, loss only on final prediction)
@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    count = 0
    for tokens, gt in dataloader:
        pred1 = model(tokens, device)[:, -1, 0]
        tokens_with_pred1 = [seq + [f"{pred1[i].item():.3f}"] for i, seq in enumerate(tokens)]
        pred2 = model(tokens_with_pred1, device)[:, -1, 0]
        loss = loss_fn(pred2.to(device), gt[:, 1].to(device))  # only compare final squared output
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / count

# Train loop (GT feeding for both stages, supervise both predictions)
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = AvgDataset("id_training_data.jsonl")
    test_set_id = AvgDataset("id_testing_data.jsonl")
    test_set_ood = AvgDataset("ood_testing_data_shift0.jsonl")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader_id = DataLoader(test_set_id, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader_ood = DataLoader(test_set_ood, batch_size=32, shuffle=False, collate_fn=collate_fn)

    config = GPT2Config(n_embd=64, n_layer=8, n_head=4, vocab_size=1)
    model = GPT2AvgPredictor(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    total_steps = len(train_loader) * 2
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    mse = nn.MSELoss()
    step = 0
    best_ood_loss = float('inf')

    for epoch in range(10):
        for tokens, gt in tqdm(train_loader, desc=f"Epoch {epoch}"):
            pred1 = model(tokens, device)[:, -1, 0]  # mean prediction
            tokens_with_gt1 = [seq + [f"{gt[i][0].item():.3f}"] for i, seq in enumerate(tokens)]
            pred2 = model(tokens_with_gt1, device)[:, -1, 0]  # square prediction

            loss = mse(pred1.to(device), gt[:, 0].to(device)) + \
                   mse(pred2.to(device), gt[:, 1].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            if step % 1000 == 0:
                id_loss = evaluate(model, test_loader_id, device, mse)
                ood_loss = evaluate(model, test_loader_ood, device, mse)
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step} | LR: {lr:.6f} | ID Loss: {id_loss:.6f} | OOD Loss: {ood_loss:.6f}")

                if ood_loss < best_ood_loss:
                    best_ood_loss = ood_loss
                    torch.save(model.state_dict(), "best_model_mean_squared.pt")
                    print(f"📦 Saved best OOD model (loss={best_ood_loss:.6f})")

if __name__ == '__main__':
    train()