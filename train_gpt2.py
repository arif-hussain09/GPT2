from dataclasses import dataclass
import torch
import time 
import torch.nn as  nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel # We need this to fetch the actual pretrained weights
# -------------------------------------------------------------------------------
@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024 # position embedding size
    n_ctx: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5



# MLP class 
class MLP(nn.Module):
    def __init__(self , config:GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # 4 times the embedding size
        self.gelu = nn.GELU() # activation function
        self.c_proj = nn.Linear(4*config.n_embd , config.n_embd) # back to embedding size
        self.c_proj.NANOGPT_SCALE_INIT_ = True  # custom attribute to indicate that this layer should be initialized with the NanoGPT scheme

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

# Causal Self Attention class
class CasualSelfAttention(nn.Module):
    def __init__(self, config:GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # for key, query, value in one go(batch)
        # output projection
        self.c_proj = nn.Linear(config.n_embd ,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT_ = True  # custom attribute to indicate that this layer should be initialized with the NanoGPT scheme

        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                             .view(1, 1, config.n_ctx, config.n_ctx)) # n_ctx is the context size, which is the maximum length of the input sequence


    def forward(self , x):
        B ,T , C = x.size() # B is batch size, T is sequence length, C is embedding size
        # calculate query, key, values for all heads in batch and move head forward to be
        # the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # split the output of c_attn into 3 parts for query, key, value  

        # # hs is head size, which is embedding size divided by number of heads
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5)) # scaled dot-product attention
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # apply the causal mask
        # att = F.softmax(att, dim=-1) # softmax to get the attention weights

        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        #-----------------------------------------------------------------------------
        # Using flash attention : 
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.bias[:,:,:T,:T],
              dropout_p=self.attn_dropout.p if self.training else 0.0,
              casual=True
              )


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side    
        y = self.resid_dropout(self.c_proj(y)) # output projection and dropout
        return y

class Block(nn.Module):
    def __init__(self, config:GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # This is where they communicate with each other:-
        # as attention is a communication mechanism where the tokens lined-up in the sequence can talk to each other and share information
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # this is where they think independently:-
        # Cause in MLP the tokens are processed independently, so they don't communicate with each other, they just think independently and then share the information in the next layer
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Wrapping the our blocks in Transformer container
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            wpe=nn.Embedding(config.n_positions, config.n_embd),  # position embedding
            drop=nn.Dropout(config.embd_pdrop),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)), # model list so we can iterate over it
            ln_f=nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),  # final layer norm
        ))

        self.lm_head = nn.Linear(config.n_embd , config.vocab_size, bias=False) # Final classification layer

        # Weight sharing Scheme  : this how wte.weight get orphan and we can use it in lm_head.weight
        self.transformer.wte.weight = self.lm_head.weight # Weight tying between token embedding and final classification layer

        self.apply(self._initializer) # Initialize the weights of the model


    def _initializer(self, module):
        if isinstance(module, nn.Linear):
            std =0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT_') and module.NANOGPT_SCALE_INIT_:
                #  attn and mlp that add residual connections, we scale the initialization by 1/sqrt(2*L) where L is the number of layers. This is to prevent the variance from growing too large as we add more layers. 
                std = 0.02 / (2 * self.config.n_layer) ** 0.5   
            nn.init.normal_(module.weight ,mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization layers with ones and zeros
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # Loading the weights from the pretrained model ( HuggingFace GPT2 model ) and copying them to our model
    @classmethod
    def from_pretrained(cls, model_type: str):
        print(f"Loading weights from pretrained gpt: {model_type}")
        
        # 1. Initialize your custom model
        config = GPT2Config()
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Ignore our causal mask buffer for weight copying
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 2. Initialize HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        
        # Ignore HuggingFace specific buffers
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('.attn.masked_bias')]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('.attn.bias')]
        
        # 3. Copy weights, accounting for transposed matrices in specific layers
        transposed_weights = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed_weights):
                # Transpose weights for Linear layers that HF implements as Conv1D
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy for everything else (Embeddings, LayerNorms, biases)
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, idx):
        # idx is the input tensor of shape (B, T) where B is batch size and T is sequence length
        B , T = idx.size()
        assert T <= self.config.n_ctx, f"Cannot forward sequence of length {T}, block size is only {self.config.n_ctx}"
        # forward the GPT model itself
        token_embeddings = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # x = token + position
        position_embeddings = self.transformer.wpe(torch.arange(T, device=idx.device)) # position embeddings of shape (T, n_embd)
        x = self.transformer.drop(token_embeddings + position_embeddings) # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits


if __name__ == "__main__":
    import tiktoken
    # Conifgure the tokenizer and generation parameters
    encoder = tiktoken.get_encoding("gpt2") # Load the GPT-2 tokenizer
    num_return_sequences = 3
    max_new_tokens = 30

    # --------------------------------------------------
    # Load the pretrained GPT-2 model
    # model = GPT2Model.from_pretrained("gpt2") # Load the pretrained model from HuggingFace
    model = GPT2Model(GPT2Config()) 
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device) # Move the model to GPU if available
    model = torch.compile(model) # Compile the model for faster training and inference





    ## Loading dataset 
    path = r"C:\Users\Administrator\OneDrive\Desktop\Projects\Transformers\GPT2\Dataset\input.txt"
    with open(path , 'r', encoding='utf-8') as f:
        data = f.read()

    tokens = encoder.encode(data[:1000], disallowed_special=()) # Encode the first 1000 characters of the dataset
    B ,T  = 4 , 32
    buf = torch.tensor(tokens[:B*T + 1], dtype=torch.long)
    buf = buf.to(device)
    x = buf[:-1].view(B, T).to(device) # Input tokens
    y = buf[1:].view(B, T).to(device)  # Target tokens (next token prediction

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 ,betas=[0.9, 0.95], eps=1e-8) # AdamW optimizer

    ## Training Loop 
    for i in range(10):

        t0 = time.time()
        optimizer.zero_grad() # Zero the gradients
        logits = model(x) # Forward pass
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) # Compute the loss
        loss.backward() # Backward pass
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step() # Update the weights

        t1 = time.time()
        print(f"Step {i} - Loss: {loss.item()} - Time: {t1 - t0:.2f}s")

    import sys ; sys.exit(0) # Exit the script after training




    # --------------------------------------------------
    # Input prompt
    # --------------------------------------------------
    text = input("Enter a prompt: ")

    input_tokens = encoder.encode(
        text,
        disallowed_special=()
    )

    # Convert to tensor and add batch dimension
    input_tokens = torch.tensor(input_tokens,dtype=torch.long).unsqueeze(0)

    # Create 3 identical prompts
    x = input_tokens.repeat(num_return_sequences,1).to(device)

    prompt_length = x.shape[1]

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --------------------------------------------------
    # Generate tokens
    # --------------------------------------------------
    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Forward pass
            logits = model(x)

            # Take logits of the last token
            logits = logits[:, -1, :]

            # Convert logits -> probabilities
            prob = F.softmax(logits, dim=-1)

            # --------------------------------------------------
            # Top-k sampling
            # --------------------------------------------------
            top_prob, top_idx = torch.topk(prob,k=50,dim=-1)

            # Renormalize probabilities inside top-k
            top_prob = top_prob / top_prob.sum(dim=-1,keepdim=True)

            # Sample one token from top-k distribution
            sampled_idx = torch.multinomial(top_prob,num_samples=1)

            # Convert sampled top-k index
            # into actual vocabulary index
            next_token = top_idx.gather(-1,sampled_idx)

            # Append token to sequence
            x = torch.cat([x, next_token],dim=1)

    # --------------------------------------------------
    # Decode generated sequences
    # --------------------------------------------------
    for i in range(num_return_sequences):

        # Keep prompt + generated tokens
        tokens = x[i].tolist()

        generated_text = encoder.decode(tokens)

        print(f"\nGenerated text {i + 1}:")
        print(generated_text)