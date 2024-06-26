import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sentence_transformers
import numpy as np
import logging
from tqdm import tqdm
import faiss
import json
from evaluate import metrics_10
import os
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Model Architecture
class UserModel(nn.Module):
    def __init__(self):
        super(UserModel, self).__init__()
        self.user_mlp = nn.Linear(768, 768) 
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, user_embeddings):
        x = self.user_mlp(user_embeddings)
        x += user_embeddings
        user_embeddings = F.normalize(x, p=2, dim=1)
        return user_embeddings

class ItemModel(nn.Module):
    def __init__(self):
        super(ItemModel, self).__init__()
        self.item_mlp = nn.Linear(768, 768)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, item_embeddings):
        x = self.item_mlp(item_embeddings)
        x += item_embeddings
        item_embeddings = F.normalize(x, p=2, dim=1)
        return item_embeddings

class UserItemDataset(Dataset):
    def __init__(self, user_embeddings, item_embeddings):
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def __len__(self):
        return len(self.user_embeddings)

    def __getitem__(self, idx):
        return self.user_embeddings[idx], self.item_embeddings[idx]

def binary_cross_entropy(user_embeddings, item_embeddings):
    batch_size = user_embeddings.size(0)
    y_true = torch.eye(batch_size).to(user_embeddings.device)
    cosine_sim = torch.matmul(user_embeddings, item_embeddings.T)
    loss = F.binary_cross_entropy_with_logits(cosine_sim, y_true)
    return loss

def cosine_embedding_loss(user_embeddings, item_embeddings):
    batch_size = user_embeddings.size(0)
    cosine_sim = torch.matmul(user_embeddings, item_embeddings.T)
    labels = torch.arange(batch_size).to(cosine_sim.device)
    pos_loss = 1 - cosine_sim[labels, labels]
    neg_loss = torch.max(torch.zeros_like(cosine_sim), cosine_sim)
    neg_loss[labels, labels] = 0
    neg_loss = neg_loss.mean(dim=1)
    return pos_loss.mean(), neg_loss.mean()

# Function to run training and evaluation
def run(domain, path):
    # Loading precomputed embeddings
    domain_path = os.path.join(path, domain)
    item_profile_embeddings = torch.load(os.path.join(domain_path, 'item_profile.pth'))
    train_user_embeddings = torch.load(os.path.join(domain_path, 'train_user_behavior.pth'))
    valid_user_embeddings = torch.load(os.path.join(domain_path, 'valid_user_behavior.pth'))
    test_user_embeddings = torch.load(os.path.join(domain_path, 'test_user_behavior.pth'))
    train_item_embeddings = torch.load(os.path.join(domain_path, 'train_item_embeddings.pth'))
    valid_item_embeddings = torch.load(os.path.join(domain_path, 'valid_item_embeddings.pth'))

    print("Embeddings loaded")

    with open(os.path.join(domain_path, f'{domain}.data_maps'), 'r') as f:
        data_maps = json.load(f)

    with open(os.path.join(domain_path, 'valid_target.txt'), 'r') as f:
        valid_target = f.readlines()
    valid_target = [line.strip() for line in valid_target]
    
    with open(os.path.join(domain_path, 'test_target.txt'), 'r') as f:
        test_target = f.readlines()
    test_target = [line.strip() for line in test_target]

    dataset = UserItemDataset(train_user_embeddings, train_item_embeddings)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    user_model = UserModel().to(device)
    item_model = ItemModel().to(device)

    optimizer = torch.optim.AdamW(list(user_model.parameters()) + list(item_model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Initialize metrics storage
    Train_Loss = []
    Valid_Loss = []
    NDCG = []
    HR = []
    Train_Entropy = []
    Valid_Entropy = []
    Train_Pos_loss = []
    Valid_Pos_loss = []
    Train_Neg_loss = []
    Valid_Neg_loss = []

    best_ndcg = -1
    best_epoch = -1

    for epoch in range(50):  # Example epoch count
        user_model.train()
        item_model.train()
        epoch_loss = 0
        pos_loss = 0
        neg_loss = 0
        entropy_loss = 0
        
        for user_emb_batch, item_emb_batch in tqdm(dataloader):
            user_embeddings = user_model(user_emb_batch.to(device))
            item_embeddings = item_model(item_emb_batch.to(device))

            pos, neg = cosine_embedding_loss(user_embeddings, item_embeddings)
            loss_bce = binary_cross_entropy(user_embeddings, item_embeddings)

            total_loss = pos + neg + loss_bce

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            pos_loss += pos.item()
            neg_loss += neg.item()
            entropy_loss += loss_bce.item()

        # Validate
        user_model.eval()
        item_model.eval()
        
        with torch.no_grad():
            model_valid_user_embeddings = user_model(valid_user_embeddings)
            model_item_profile_embeddings = item_model(item_profile_embeddings)
            model_valid_item_embeddings = item_model(valid_item_embeddings)

            v_pos, v_neg = cosine_embedding_loss(model_valid_user_embeddings, model_valid_item_embeddings)
            v_loss_bce = binary_cross_entropy(model_valid_user_embeddings, model_valid_item_embeddings)
            v_total = v_pos + v_neg + v_loss_bce

            index = faiss.IndexFlatIP(768)
            index.add(model_item_profile_embeddings.cpu().detach().numpy())
            distances, indices = index.search(model_valid_user_embeddings.cpu().detach().numpy(), 10)
            predictions = []
            for (d, idx) in zip(distances, indices):
                top_10_sentences = [data_maps['id2item'][i] for i in idx]
                predictions.append(top_10_sentences)

            ndcg, hr = metrics_10(valid_target, predictions,10)

        avg_epoch_loss = epoch_loss / len(dataloader)
        Train_Loss.append(avg_epoch_loss)
        Valid_Loss.append(v_total.item())
        NDCG.append(ndcg)
        HR.append(hr)
        Train_Entropy.append(entropy_loss / len(dataloader))
        Valid_Entropy.append(v_loss_bce.item())
        Train_Pos_loss.append(pos_loss / len(dataloader))
        Valid_Pos_loss.append(v_pos.item())
        Train_Neg_loss.append(neg_loss / len(dataloader))
        Valid_Neg_loss.append(v_neg.item())

        print(f"Epoch: {epoch}, Train Loss: {avg_epoch_loss:.4f}, Valid Loss: {v_total:.4f}, NDCG: {ndcg:.4f}, HR: {hr:.4f}, "
              f"Train_Entropy: {entropy_loss/len(dataloader):.4f}, Train_Pos_loss: {pos_loss/len(dataloader):.4f}, "
              f"Train_Neg_loss: {neg_loss/len(dataloader):.4f}, Valid_Entropy: {v_loss_bce:.4f}, "
              f"Valid_Pos_loss: {v_pos:.4f}, Valid_Neg_loss: {v_neg:.4f}")

        # Save best model based on NDCG@10
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch
            save_dir = os.path.join(domain_path, f'best_model.pth')
            torch.save({
                'epoch': epoch,
                'user_model_state_dict': user_model.state_dict(),
                'item_model_state_dict': item_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'ndcg': ndcg,
                'hr': hr
            }, save_dir)

        scheduler.step(avg_epoch_loss)

    # Plotting metrics
    Epoch = list(range(50))
    fig, axs = plt.subplots(6, 1, figsize=(10, 20), sharex=True)

    axs[0].plot(Epoch, Train_Loss, label='Train Loss', marker='o')
    axs[0].plot(Epoch, Valid_Loss, label='Valid Loss', marker='o')
    axs[0].set_title('Total Loss')
    axs[0].legend()

    axs[1].plot(Epoch, NDCG, label='NDCG', marker='o')
    axs[1].set_title('NDCG')
    axs[1].legend()

    axs[2].plot(Epoch, HR, label='HR', marker='o')
    axs[2].set_title('HR')
    axs[2].legend()

    axs[3].plot(Epoch, Train_Entropy, label='Train Entropy', marker='o')
    axs[3].plot(Epoch, Valid_Entropy, label='Valid Entropy', marker='o')
    axs[3].set_title('Entropy')
    axs[3].legend()

    axs[4].plot(Epoch, Train_Pos_loss, label='Train Positive Loss', marker='o')
    axs[4].plot(Epoch, Valid_Pos_loss, label='Valid Positive Loss', marker='o')
    axs[4].set_title('Positive Loss')
    axs[4].legend()

    axs[5].plot(Epoch, Train_Neg_loss, label='Train Negative Loss', marker='o')
    axs[5].plot(Epoch, Valid_Neg_loss, label='Valid Negative Loss', marker='o')
    axs[5].set_title('Negative Loss')
    axs[5].legend()

    plt.show()

    # Testing
    checkpoint = torch.load(os.path.join(domain_path, 'best_model.pth'))
    user_model.load_state_dict(checkpoint['user_model_state_dict'])
    item_model.load_state_dict(checkpoint['item_model_state_dict'])

    user_model.eval()
    item_model.eval()

    with torch.no_grad():
        model_test_user_embeddings = user_model(test_user_embeddings.to(device))
        model_item_profile_embeddings = item_model(item_profile_embeddings.to(device))

        index = faiss.IndexFlatIP(768)
        index.add(model_item_profile_embeddings.cpu().detach().numpy())
        distances, indices = index.search(model_test_user_embeddings.cpu().detach().numpy(), 50)
        predictions = []
        for (d, idx) in zip(distances, indices):
            top_10_sentences = [data_maps['id2item'][i] for i in idx]
            predictions.append(top_10_sentences)

        for k in [10, 50]:
            print(f'For test set')
            ndcg, hr = metrics_10(test_target, predictions, k)
            print(f'NDCG@{k}: {ndcg:.4f}')
            print(f'HR@{k}: {hr:.4f}')

# Assign arguments directly
domain = 'All_Beauty'
path = 'processed'

print(f'LLM + MLP for {domain}')
run(domain, path)
