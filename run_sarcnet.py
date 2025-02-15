import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import gc
from timm.layers import PatchEmbed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from sklearn.cluster import KMeans
import random
from torch.cuda.amp import GradScaler, autocast
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from peft import LoraConfig,LoraModel
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

llm_path="/your/llm/path"



class MultiDataset(Dataset):
    def __init__(self, X_image_path,X_text,Y,resize, train_mode=True):

        self.image_path = X_image_path
        self.text=X_text
        self.labels = torch.tensor(Y)
        self.resize = resize
        self.train_mode = train_mode  # 添加训练模式标志
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        
        img=Image.open("./SarcNet Image-Text/Image/"+self.image_path[idx]).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((self.resize, self.resize))])
        image_tensor = transform(img).float()
        
        return image_tensor,self.text[idx], self.labels[idx]
from torch import Tensor
class Cross_attention(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.01):
        super(Cross_attention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bm=nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        config = LoraConfig(
            r=32, lora_alpha=16, lora_dropout=0.01, bias="none", task_type="SEQ_CLS",target_modules=["query_key_value","dense"]
        )
        self.llm_model=AutoModelForSequenceClassification.from_pretrained(llm_path,_attn_implementation="flash_attention_2",
                                                                          torch_dtype=torch.bfloat16, num_labels=1,classifier_dropout=0.01,trust_remote_code=True)
        self.llm_model = LoraModel(self.llm_model, config, "default")
        self.tokenizer=AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # for param in self.llm_model.parameters():
        #     param.requires_grad = False
        # torch.nn.init.xavier_uniform_(self.llm_model.classifier_head1.weight)
        # torch.nn.init.xavier_uniform_(self.llm_model.classifier_head2.weight)
        # torch.nn.init.kaiming_uniform_(self.llm_model.classifier_head1.weight)
        # torch.nn.init.kaiming_uniform_(self.llm_model.classifier_head2.weight)
        # torch.nn.init.xavier_uniform_(self.llm_model.classifier_head3.weight)
        
        torch.nn.init.uniform_(self.llm_model.classifier_head1.weight, a=-1e-2, b=1e-2)
        torch.nn.init.uniform_(self.llm_model.classifier_head2.weight, a=-1e-2, b=1e-2)
        torch.nn.init.uniform_(self.llm_model.classifier_head3.weight, a=-1e-2, b=1e-2)
        torch.nn.init.uniform_(self.llm_model.bm1.weight, a=-5,b=5)
        torch.nn.init.uniform_(self.llm_model.bm2.weight, a=-5,b=5)
        torch.nn.init.uniform_(self.llm_model.bm3.weight, a=-5,b=5)
        nn.init.constant_(self.llm_model.classifier_head1.bias,0)
        nn.init.constant_(self.llm_model.classifier_head2.bias,0)
        nn.init.constant_(self.llm_model.classifier_head3.bias,0)
        nn.init.constant_(self.llm_model.bm1.bias,0)
        nn.init.constant_(self.llm_model.bm2.bias,0)
        nn.init.constant_(self.llm_model.bm3.bias,0)
        # print(self.llm_model.bm1.bias)
        # print(self.llm_model.classifier_head1.bias)
        
        self.llm_model.bm1.bias.requires_grad =True
        self.llm_model.bm1.weight.requires_grad =True
        self.llm_model.classifier_head1.requires_grad =True
        self.llm_model.classifier_head1.weight.requires_grad =True
        self.llm_model.classifier_head1.bias.requires_grad =True
        
        self.llm_model.bm2.bias.requires_grad =True
        self.llm_model.bm2.weight.requires_grad =True
        self.llm_model.classifier_head2.requires_grad =True
        self.llm_model.classifier_head2.weight.requires_grad =True
        self.llm_model.classifier_head2.bias.requires_grad =True
        
        self.llm_model.bm3.bias.requires_grad =True
        self.llm_model.bm3.weight.requires_grad =True
        self.llm_model.classifier_head3.requires_grad =True
        self.llm_model.classifier_head3.weight.requires_grad =True
        self.llm_model.classifier_head3.bias.requires_grad =True
        self.patch_embedding = PatchEmbed(
            img_size=512,
            patch_size=14,
            in_chans=3,
            embed_dim=4096, #和GLM4大小一致
            flatten=True,
        )
        self.word_embeddings = self.llm_model.transformer.embedding.word_embeddings.weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.cross_atten= Cross_attention(4096, 8, 32, 4096)
        self.patch_nums = int((224 - 3) / 8 + 2)
        self.head_nf = 32 * self.patch_nums
    def freeze_llm(self):
        for param in self.llm_model.parameters():
            param.requires_grad = False
    def forward(self, x_enc, prompt):
        x_enc = self.bm(x_enc)
        prompt = self.tokenizer(prompt, return_tensors="pt",padding=True,truncation=True, max_length=2048)
        mask=prompt.attention_mask
        posid=prompt.position_ids
        prompt=prompt.input_ids

        prompt_embeddings = self.llm_model.transformer.embedding.word_embeddings(prompt.to(x_enc.device)).requires_grad_(True)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0).float()).permute(1, 0)
        
        #print(x_enc.shape)
        n_vars=x_enc.shape[1]
        enc_out = self.patch_embedding(x_enc).requires_grad_(True)
        #enc_out = F.normalize(enc_out, dim=1)
        #print(enc_out.shape)
        enc_out = self.cross_atten(enc_out, source_embeddings, source_embeddings).to(torch.bfloat16)
        # print(prompt_embeddings.shape)
        St,Ed= torch.split(prompt_embeddings,[prompt_embeddings.shape[1]-1,1],dim=1)
        llm_enc_out = torch.cat([St, enc_out,Ed], dim=1)
        #print(llama_enc_out.shape)
        B,T,_=enc_out.shape
        # print(mask.shape)
        # print(torch.ones([B,T]).shape)
        att_mask=torch.cat([mask, torch.ones([B,T])], dim=1).to(self.llm_model.device)
        #print(att_mask.shape)
        x,y=posid.shape
        tens=torch.tensor([[]]).to(self.llm_model.device)
        for i in range(x):
            if i==0:
                tens=torch.arange(posid[i][y-1]+1,posid[i][y-1]+1+T,1).view(1,-1)
            else:
                # print(tens.shape)
                # print(torch.arange(posid[i][y-1]+1,posid[i][y-1]+T+1,1).view(1,-1).shape)
                # print(torch.arange(posid[i][y-1]+1,posid[i][y-1]+T+1,1).view(1,-1))
                # print(posid)
                tens=torch.cat([tens,torch.arange(posid[i][y-1]+1,posid[i][y-1]+1+T,1).view(1,-1)],dim=0)
        #print(posid.shape)
        #print(tens.shape)
        pos_ids=torch.cat([posid, tens], dim=1).to(self.llm_model.device)
        # print(llm_enc_out.shape)
        # print(att_mask.shape)
        # print(pos_ids.shape)
        dec_out = self.llm_model(inputs_embeds=llm_enc_out,attention_mask=att_mask,position_ids=pos_ids)
        # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        # dec_out = dec_out.permute(0, 2, 1).contiguous()
        # dec_out = self.normalize_layers(dec_out, 'denorm')
        return F.sigmoid(dec_out.logits)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
      
        
    
df_train = pd.read_csv("./SarcNet Image-Text/SarcNetTrain.csv", encoding='gbk')
df_test = pd.read_csv("./SarcNet Image-Text/SarcNetTest.csv", encoding='gbk')
df_val = pd.read_csv("./SarcNet Image-Text/SarcNetVal.csv", encoding='gbk')



train_dataset = MultiDataset(df_train['Imagepath'].tolist(),df_train['Text'].tolist(),df_train['Multi_label'].tolist(), resize=512, train_mode=True)
test_dataset = MultiDataset(df_test['Imagepath'].tolist(),df_test['Text'].tolist(),df_test['Multi_label'].tolist(), resize=512, train_mode=False)
val_dataset = MultiDataset(df_val['Imagepath'].tolist(),df_val['Text'].tolist(),df_val['Multi_label'].tolist(), resize=512, train_mode=False)
    
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)

accelerator = Accelerator()
trained_parameters = []
model=Model()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {num_learnable_params}")

#criterion = nn.CrossEntropyLoss(weight= torch.tensor([437.0/72.0/4,437.0/53.0/4,1.0/4.0,437.0/24.0/4]).to(accelerator.device))
criterion = nn.BCELoss()
for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)
optim = torch.optim.Adam(trained_parameters, lr=2e-5)
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=10*len(train_loader), eta_min=1e-8)
#scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,steps_per_epoch=train_steps,pct_start=args.pct_start,epochs=args.train_epochs,max_lr=args.learning_rate)
train_loader,val_loader, test_loader,  model, model_optim, scheduler = accelerator.prepare(
        train_loader,val_loader, test_loader, model, optim, scheduler)
from tqdm import tqdm
# scaler = torch.cuda.amp.GradScaler()
#f=open("/root/pam.txt","w")
from sklearn.metrics import classification_report
f=open("res.txt","w")
for epoch in range(0,10):
    torch.cuda.empty_cache()
    gc.collect()
    model.train()
    with tqdm(total=len(train_loader)) as pbar:
        sloss=0.0
        for i, (inputs,prompt, labels) in enumerate(train_loader):
            # torch.cuda.empty_cache()
            # gc.collect()
            inputs=inputs.to(accelerator.device)
            labels=labels.to(accelerator.device)
            outputs = model(inputs,prompt)
            #print(outputs, labels.float())
            loss = criterion(outputs.float().cuda(), labels.float().view(outputs.shape).cuda())
            sloss+=loss

            #print(loss)
            #print(model_optim.state_dict()['param_groups'][0]['lr'])
            #next=model.llm_model.classifier_head.weight.clone()
            # for name, parms in model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     f.write('-->name:')
            #     f.write(name)
            #     f.write("\n")
            #     f.write('-->para:')
            #     f.write(str(parms))
            #     f.write("\n")
            #     f.write('-->grad_requirs:')
            #     f.write(str(parms.requires_grad))
            #     f.write("\n")
            #     f.write('-->grad_value:')
            #     f.write(str(parms.grad))
            #     f.write("\n")
            #     f.write("===")
            #     f.write("\n")
            #     print("===")
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()
            # scaler.update()
            scheduler.step()
            #f.flush()
            #print(torch.equal(next,model.llm_model.classifier_head.weight))
            pbar.update(1)
        print(sloss/len(train_loader))
        print(model_optim.state_dict()['param_groups'][0]['lr'], file=f, flush=True)
        print(sloss/len(train_loader), file=f, flush=True)
    if epoch%1==0:
        model.eval()
        pre_true=[]
        pre=[]
        with tqdm(total=len(val_loader)) as pbar:
            for i, (inputs,prompt, labels) in enumerate(val_loader):
                # torch.cuda.empty_cache()
                # gc.collect()
                inputs=inputs.to(accelerator.device)
                labels=labels.to(accelerator.device)
                outputs = model(inputs,prompt)
                # print(outputs, labels)
                #loss = criterion(outputs, labels.float())
                pre_true.extend(labels.view((-1)).tolist())
                # print(prompt, file=f, flush=True)
                # print(torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs)).int().view(labels.shape).view((-1)).tolist(), file=f, flush=True)
                
                pre.extend(torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs)).int().view(labels.shape).view((-1)).tolist())
                pbar.update(1)
        # print(pre,pre_true)
        print(classification_report(pre_true,pre,digits=4))
        print(classification_report(pre_true,pre,digits=4), file=f, flush=True)
model.eval()
pre_true=[]
pre=[]
with tqdm(total=len(test_loader)) as pbar:
    for i, (inputs,prompt, labels) in enumerate(test_loader):
        # torch.cuda.empty_cache()
        # gc.collect()
        inputs=inputs.to(accelerator.device)
        labels=labels.to(accelerator.device)
        outputs = model(inputs,prompt)
        # print(outputs, labels)
        #loss = criterion(outputs, labels.float())
        pre_true.extend(labels.view((-1)).tolist())
                
        pre.extend(torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs)).int().view(labels.shape).view((-1)).tolist())
        pbar.update(1)
    print(classification_report(pre_true,pre,digits=4))
    print(classification_report(pre_true,pre,digits=4), file=f, flush=True)         
f.close()