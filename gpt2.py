from dataclasses import dataclass
import torch 
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
import math
import inspect
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import  torch.distributed as  dist 

import numpy as np

import time 
class MLP(nn.Module):
    
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.gpt_scale = 1 
        self.act = nn.GELU(approximate= 'tanh')
    
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
   
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.gpt_scale = 1 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B , T , C = x.size()
        qkv = self.c_attn(x)
        q , k , v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        
        
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head))   
        # att = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        
        y = F.scaled_dot_product_attention(q, k, v , is_causal=True) 
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    
    
      
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn =  CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPT2Config:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768
    



class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2, self).__init__()
        self.config = config
    
    
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
           
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self.__init_weights)
        
    
    def __init_weights(self , module):
        
  
        
        
        if isinstance(module , nn.Linear):
            std = 0.02
            if hasattr(module , 'gpt_scale'):
                
                std *= (2 * self.config.n_layer) ** -0.5
                
                
            torch.nn.init.normal_(module.weight ,  mean = 0.0 , std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module , nn.Embedding):
            torch.nn.init.normal_(module.weight , mean = 0.0 , std=0.02)
        
        
    def forward(self, idx , targets=None):
        B , T = idx.size()
        
        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size}, but input has {T} tokens."
        
        token_embeddings = self.transformer.wte(idx)
        
        position_embeddings = self.transformer.wpe(torch.arange(T, device=idx.device))
        
        x = token_embeddings + position_embeddings
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
         
        return logits , loss
        

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], f"Model type {model_type} not supported"
        from transformers import GPT2LMHeadModel
        print("Loading pretrained model from HuggingFace")
        
        config_args = {
                'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
         

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        model_hf_sd = model_hf.state_dict()
        
        sd_keys_hf = model_hf_sd.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys) == len(sd_keys_hf), f"Mismatch in number of keys: {len(sd_keys)} vs {len(sd_keys_hf)}"
       
       
        for k in sd_keys_hf:
            if any([k.endswith(t) for t in transposed]):
                assert model_hf_sd[k].shape[::-1] == sd[k].shape
                
                with torch.no_grad():
                    sd[k].copy_(model_hf_sd[k].T)
            else:
                
                assert model_hf_sd[k].shape == sd[k].shape, f"Shape mismatch for key {k}: {model_hf_sd[k].shape} vs {sd[k].shape}"
                
                with torch.no_grad():
                    sd[k].copy_(model_hf_sd[k])
        
        return model
        
        
    def configure_optimizers(self, weight_decay, lr , device):
        
        param_dict = {pn : p for pn, p in self.named_parameters() }
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}
        
        
        decay_params = [p for n , p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n , p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        
        if master_process:
            print(f"Using fused AdamW: {use_fused}")
            
            
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    
    
        
        
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
   
ddp =  int(os.environ.get('RANK' , -1)) != -1
if ddp:
    
    assert torch.cuda.is_available(), "Distributed training requires CUDA"
    
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    #print(f"ddp_rank: {ddp_rank}")
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    #print(f"ddp_world_size: {ddp_world_size}")
    #print(f"dpp_local_rank: {ddp_local_rank}")
    
    device = f'cuda:{ddp_local_rank}'
    print(f"Using device: {device}")
 
    
    torch.cuda.set_device(device)
    master_process = ddp_local_rank == 0
else : 
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    print("Running in single GPU mode")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    


# -------------------------------------------------------------------------#
#                            Loading the dataset                           #
# -------------------------------------------------------------------------#

def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt
    

class DataLoaderLite : 
    def __init__(self, B , T , process_rank , num_process , split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        assert split in {'train' , 'valid'}
        
        data_root = 'edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [f for f in shards if f.endswith('.npy')]
        shards = sorted(shards)
        shards = [os.path.join(data_root , f) for f in shards]
        self.shards = shards
        
        assert len(shards) > 0, "No shards found in the data root"
        
        if master_process:
            print(f"Found {len(shards)} shards in the data root")
            
        self.reset()
        
        
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        
        
        self.current_position = self.B*self.T * self.process_rank       
    def next_batch(self):
        B , T = self.B , self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        
        x = buf[:-1].view(B , T)
        y = buf[1:].view(B , T)
        self.current_position += B * T * self.num_process
        if self.current_position + B * T * self.num_process >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x , y




torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
   
total_batch_size = 524288   
B = 16
T = 1024

assert total_batch_size % (B * T * ddp_world_size) == 0, f"Total batch size {total_batch_size} must be divisible by B * T * ddp_world_size"

grad_accumulation_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"Batch size: {B} | Sequence length: {T} | Total batch size: {total_batch_size}")
    print(f"Gradient accumulation steps: {grad_accumulation_steps}")

train_dataloader = DataLoaderLite(B, T , process_rank = ddp_rank, num_process = ddp_world_size , split = 'train') 
val_dataloader = DataLoaderLite(B, T , process_rank = ddp_rank, num_process = ddp_world_size , split = 'valid')

#torch.set_float32_matmul_precision('high')

model = GPT2(GPT2Config(vocab_size=50304))

if master_process:
    print("Loaded model successfully")

model = model.to(device)
if master_process:
    print("Moved model to device successfully")

model = torch.compile(model) 
    


if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])

raw_model = model.module if ddp else model
  

# Parameters count 

params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if master_process:
    print(f"Total parameters: {params / 1e6:.2f}M")


max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073



def get_lr(step):  
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr 
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"Decay ratio out of bounds: {decay_ratio}"
    
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * math.pi)))
    return min_lr + (max_lr - min_lr) * coeff
    
optimizer = raw_model.configure_optimizers(weight_decay=0.1, lr=6e4, device = device)

for step in range(max_steps):
    
    t0 = time.time()
    
    if step % 100 == 0 :
        model.eval()
        val_dataloader.reset()
        
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x , y = val_dataloader.next_batch()
                x = x.to(device)
                y = y.to(device)
                
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, val_loss = model(x , y)
                val_loss = val_loss / val_loss_steps
                val_loss_accum += val_loss.detach()
                
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
                val_loss_accum = val_loss_accum / ddp_world_size
            
            if master_process:
                print(f"Validation loss: {val_loss_accum.item():.4f}")

        
    
    
    
    
    # training loop 
    
    model.train()
   
    optimizer.zero_grad()
    # 1) make loss_accum a tensor on the right device
    loss_accum = torch.zeros([], device=device)
    for micro_step in range(grad_accumulation_steps):
            
        x , y = train_dataloader.next_batch()
        x = x.to(device)
        y = y.to(device)    
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
         
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits , loss = model(x , y)
        loss = loss / grad_accumulation_steps  
        loss_accum += loss.detach() 
           
            
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
        loss_accum = loss_accum / ddp_world_size
          
    norm = torch.nn.utils.clip_grad_norm_(model.parameters() , 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step()
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_dataloader.B * train_dataloader.T * grad_accumulation_steps * ddp_world_size
    
    tokens_per_second = tokens_processed / dt
    
    
    if master_process:
        print(f"Step {step}: Loss = {loss_accum.item():.6f} | Time = {(t1 - t0) * 1000:.2f}ms | Tokens per second = {tokens_per_second:.2f}  |  Norm = {norm:.4f} | lr : {lr:.4e}")


if ddp:
    destroy_process_group()


