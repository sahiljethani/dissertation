
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
import argparse



print(torch.cuda.is_available())  # Python 3.x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Model Architecture
def run(domain,path):
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
            x+=user_embeddings
            user_embeddings = F.normalize(x, p=2, dim=1)
            return user_embeddings


    class ItemModel(nn.Module):
        def __init__(self):
            super(ItemModel, self).__init__()

            self.item_mlp =nn.Linear(768, 768)
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.eye_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, item_embeddings):

            x = self.item_mlp(item_embeddings)
            x+=item_embeddings
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
        
        
    #Definig the loss function
        
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
        return pos_loss.mean(),neg_loss.mean()




    #Training

    #PRECOMPUTED EMBEDDINGS



    print("Loading precomputed embeddings")

    domain_path=os.path.join(path, domain)


    item_profile_embeddings = torch.load(os.path.join(domain_path,'item_profile.pth'))
    train_user_embeddings = torch.load(os.path.join(domain_path,'train_user_behavior.pth'))
    valid_user_embeddings = torch.load(os.path.join(domain_path,'valid_user_behavior.pth'))
    test_user_embeddings = torch.load(os.path.join(domain_path,'test_user_behavior.pth'))
    train_item_embeddings = torch.load(os.path.join(domain_path,'train_item_embeddings.pth'))
    valid_item_embeddings = torch.load(os.path.join(domain_path,'valid_item_embeddings.pth'))

    print("Embeddings loaded")




    with open(os.path.join(domain_path,f'{domain}.data_maps'), 'r') as f:
        data_maps = json.load(f)



    # valid_item_texts = []
    with open(os.path.join(domain_path,'valid_target.txt'), 'r') as f:
        valid_target = f.readlines()

    with open(os.path.join(domain_path,'test_target.txt'), 'r') as f:
        test_target = f.readlines()
    test_target = [line.strip() for line in test_target]





    dataset = UserItemDataset(train_user_embeddings, train_item_embeddings)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)



    user_model = UserModel().to(device)
    item_model = ItemModel().to(device)


    optimizer = torch.optim.AdamW(list(user_model.parameters()) + list(item_model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

    for epoch in range(100):  # Example epoch count
        user_model.train()
        item_model.train()
        epoch_loss = 0
        pos_loss=0
        neg_loss=0
        entropy_loss=0
        
        for user_emb_batch, item_emb_batch in tqdm(dataloader):

            user_embeddings = user_model(user_emb_batch)
            item_embeddings = item_model(item_emb_batch)

            pos,neg = cosine_embedding_loss(user_embeddings, item_embeddings)
            loss_bce = binary_cross_entropy(user_embeddings, item_embeddings)

            
            total_loss = pos+2*neg+loss_bce

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            pos_loss+=pos
            neg_loss+=neg
            entropy_loss+=loss_bce
            
            

        #valid
        user_model.eval()
        item_model.eval()
        
        with torch.no_grad():
            model_valid_user_embeddings = user_model(valid_user_embeddings)
            model_item_profile_embeddings = item_model(item_profile_embeddings)
            model_valid_item_embeddings = item_model(valid_item_embeddings)

            
            
            v_pos,v_neg = cosine_embedding_loss(model_valid_user_embeddings, model_valid_item_embeddings)
            v_loss_bce = binary_cross_entropy(model_valid_user_embeddings, model_valid_item_embeddings)
            v_total=v_pos+v_neg+v_loss_bce
            
            index = faiss.IndexFlatIP(768)
            index.add(model_item_profile_embeddings.cpu().detach().numpy())
            distances, indices = index.search(model_valid_user_embeddings.cpu().detach().numpy(), 10)
            predictions = []
            for (d, idx) in zip(distances, indices):
                top_10_sentences = [data_maps['id2item'][i] for i in idx]
                predictions.append(top_10_sentences)

            ndcg, hr = metrics_10(valid_target, predictions)

        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch: {epoch}, Train Loss: {avg_epoch_loss:.4f},Valid Loss: {v_total:.4f}, NDCG: {ndcg:.4f}, HR: {hr:.4f},Train_Entropy: {entropy_loss/len(dataloader):.4f},Train_Pos_loss: {pos_loss/len(dataloader):.4f},Train_Neg_loss:{neg_loss/len(dataloader):.4f},Valid_Entropy: {v_loss_bce:.4f},valid_Pos_loss: {v_pos:.4f},valid_neg_loss: {v_neg:.4f}")
        
        save_dir = os.path.join(domain_path,f'model_mlp/model_{epoch}.pth')

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


    #TESTING
        
    checkpoint = torch.load('/kaggle/working/model_23.pth')
    user_model.load_state_dict(checkpoint['user_model_state_dict'])
    item_model.load_state_dict(checkpoint['item_model_state_dict'])


    user_model.eval()
    item_model.eval()


    with torch.no_grad():
        model_test_user_embeddings = user_model(test_user_embeddings)
        model_item_profile_embeddings = item_model(item_profile_embeddings)

        index = faiss.IndexFlatIP(768)
        index.add(model_item_profile_embeddings.cpu().detach().numpy())
        distances, indices = index.search(model_test_user_embeddings.cpu().detach().numpy(), 50)
        predictions = []
        for (d, idx) in zip(distances, indices):
            top_10_sentences = [data_maps['id2item'][i] for i in idx]
            predictions.append(top_10_sentences)

        k=[10,50]
        for k in k:
            print(f'For test set')
            ndcg, hr = metrics_10(test_target, predictions,k)
            print(f'NDCG@{k}: {ndcg:.4f}')
            print(f'HR@{k}: {hr:.4f}')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='All_Beauty', help='dataset name')
    parser.add_argument('--path', type=str, default='processed', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    print(f'LLM + MLP for {args.domain}')

    run(args.domain,args.path)
