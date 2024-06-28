
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from tqdm import tqdm
import faiss
import json
from evaluate import metrics_10
import os
import argparse
import matplotlib.pyplot as plt




print(torch.cuda.is_available())  # Python 3.x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Model Architecture
def run(domain,path,batch=512,n_epochs=10):
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
    valid_target = [line.strip() for line in valid_target]

    with open(os.path.join(domain_path,'test_target.txt'), 'r') as f:
        test_target = f.readlines()
    test_target = [line.strip() for line in test_target]



    dataset = UserItemDataset(train_user_embeddings, train_item_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)



    user_model = UserModel().to(device)
    item_model = ItemModel().to(device)


    optimizer = torch.optim.AdamW(list(user_model.parameters()) + list(item_model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)


    # Initialize metrics storage
    Train_Loss, Valid_Loss, NDCG, HR = [], [], [], []
    Train_Entropy, Valid_Entropy, Train_Pos_loss, Valid_Pos_loss = [], [], [], []
    Train_Neg_loss, Valid_Neg_loss = [], []

    best_ndcg = -1
    best_hr = -1


    print("Training started")

    for epoch in range(n_epochs):  # Example epoch count
        user_model.train()
        item_model.train()

        epoch_loss, pos_loss, neg_loss, entropy_loss = 0, 0, 0, 0
        
        for user_emb_batch, item_emb_batch in tqdm(dataloader):

            user_embeddings = user_model(user_emb_batch.to(device))
            item_embeddings = item_model(item_emb_batch.to(device))

            pos,neg = cosine_embedding_loss(user_embeddings, item_embeddings)
            loss_bce = binary_cross_entropy(user_embeddings, item_embeddings)

            
            total_loss = pos+2*neg+loss_bce

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            pos_loss+=pos.item()
            neg_loss+=neg.item()
            entropy_loss+=loss_bce.item()
            
            

        #valid
        user_model.eval()
        item_model.eval()
        valid_loss, valid_pos_loss, valid_neg_loss, valid_entropy_loss = 0, 0, 0, 0
        valid_dataloader = DataLoader(UserItemDataset(valid_user_embeddings, valid_item_embeddings), batch_size=batch // 2, shuffle=False)

    
        with torch.no_grad():
            for v_user_emb_batch, v_item_emb_batch in tqdm(valid_dataloader):
                v_user_embeddings = user_model(v_user_emb_batch)
                v_item_embeddings = item_model(v_item_emb_batch)

                v_pos, v_neg = cosine_embedding_loss(v_user_embeddings, v_item_embeddings)
                v_loss_bce = binary_cross_entropy(v_user_embeddings, v_item_embeddings)
                v_total = v_pos + 2 * v_neg + v_loss_bce

                valid_loss += v_total.item()
                valid_pos_loss += v_pos.item()
                valid_neg_loss += v_neg.item()
                valid_entropy_loss += v_loss_bce.item()

            valid_loss /= len(valid_dataloader)
            valid_pos_loss /= len(valid_dataloader)
            valid_neg_loss /= len(valid_dataloader)
            valid_entropy_loss /= len(valid_dataloader)
        
            model_valid_user_embeddings = user_model(valid_user_embeddings)
            model_item_profile_embeddings = item_model(item_profile_embeddings)

            
            index = faiss.IndexFlatIP(768)
            gpu_index = faiss.index_cpu_to_all_gpus(index)
            gpu_index.add(model_item_profile_embeddings.cpu().detach().numpy())
            distances, indices = gpu_index.search(model_valid_user_embeddings.cpu().detach().numpy(), 10)
            # index.add(model_item_profile_embeddings.cpu().detach().numpy())
            # distances, indices = index.search(model_valid_user_embeddings.cpu().detach().numpy(), 10)
            predictions = []
            for (d, idx) in zip(distances, indices):
                top_10_sentences = [data_maps['id2item'][i] for i in idx]
                predictions.append(top_10_sentences)

            ndcg, hr = metrics_10(valid_target, predictions,10)

        
        Train_Loss.append(epoch_loss / len(dataloader))
        Valid_Loss.append(valid_loss)
        NDCG.append(ndcg)
        HR.append(hr)
        Train_Entropy.append(entropy_loss / len(dataloader))
        Valid_Entropy.append(valid_entropy_loss)
        Train_Pos_loss.append(pos_loss / len(dataloader))
        Valid_Pos_loss.append(valid_pos_loss)
        Train_Neg_loss.append(neg_loss / len(dataloader))
        Valid_Neg_loss.append(valid_neg_loss)

        print(f"Epoch: {epoch}, Train Loss: {epoch_loss / len(dataloader):.4f}, Valid Loss: {valid_loss:.4f}, NDCG: {ndcg:.4f}, HR: {hr:.4f}, "
              f"Train_Entropy: {entropy_loss / len(dataloader):.4f}, Train_Pos_loss: {pos_loss / len(dataloader):.4f}, "
              f"Train_Neg_loss: {neg_loss / len(dataloader):.4f}, Valid_Entropy: {valid_entropy_loss:.4f}, "
              f"Valid_Pos_loss: {valid_pos_loss:.4f}, Valid_Neg_loss: {valid_neg_loss:.4f}")
        
        # Save best model based on NDCG@10
        if ndcg > best_ndcg or (ndcg == best_ndcg and hr > best_hr):
            best_ndcg = ndcg
            best_hr=hr
            
            save_dir = os.path.join(domain_path, f'{domain}_best_model.pth')
            torch.save({
                'epoch': epoch,
                'user_model_state_dict': user_model.state_dict(),
                'item_model_state_dict': item_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader),
                'ndcg': ndcg,
                'hr': hr
            }, save_dir)
        
    

        scheduler.step(epoch_loss / len(dataloader))

    print("Training completed")

     # Plotting metrics
    Epoch = list(range(n_epochs))
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

    #plt.show()
    fig.savefig(os.path.join(domain_path, f'{domain}_loss_plot.png'))

    print("Metrics plotted")
    


    #TESTING

    print("Testing started")
        
    checkpoint = torch.load(os.path.join(domain_path, f'{domain}_best_model.pth'))
    user_model.load_state_dict(checkpoint['user_model_state_dict'])
    item_model.load_state_dict(checkpoint['item_model_state_dict'])


    user_model.eval().to(device)
    item_model.eval().to(device)


    with torch.no_grad():
        model_test_user_embeddings = user_model(test_user_embeddings)
        model_item_profile_embeddings = item_model(item_profile_embeddings)

        index = faiss.IndexFlatIP(768)
        index.add(model_item_profile_embeddings.cpu().detach().numpy())           
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        gpu_index.add(model_item_profile_embeddings.cpu().detach().numpy())
        distances, indices = gpu_index.search(model_valid_user_embeddings.cpu().detach().numpy(), 50)
        # distances, indices = index.search(model_test_user_embeddings.cpu().detach().numpy(), 50)
        predictions = []
        for (d, idx) in zip(distances, indices):
            top_10_sentences = [data_maps['id2item'][i] for i in idx]
            predictions.append(top_10_sentences)

        for k in [10, 50]:
            print(f'For test set')
            ndcg, hr = metrics_10(test_target, predictions, k)
            print(f'NDCG@{k}: {ndcg:.4f}')
            print(f'HR@{k}: {hr:.4f}')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='All_Beauty', help='dataset name')
    parser.add_argument('--path', type=str, default='processed', help='path to dataset')
    parser.add_argument('--batch', type=int, default=512, help='batch size')
    args, unparsed = parser.parse_known_args()
    print(args)

    print(f'LLM + MLP for {args.domain}')

    run(args.domain,args.path,args.batch)
