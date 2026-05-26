import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    """
    A standard 1D Convolutional block to extract local, time-dependent info.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape: [Batch, Length, Dim]
        # Conv1d expects [Batch, Dim, Length]
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        return x + residual # Using a residual connection

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, target_dim, num_conv_blocks=3):
        super().__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Fully-connected projection layers
        self.project_to_target = nn.Linear(embed_dim, target_dim)
        
        # 3. Series of convolution blocks (Section 3.3)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(target_dim) for _ in range(num_conv_blocks)
        ])
        
        # 4. Project back to embedding dimension
        self.project_to_embed = nn.Linear(target_dim, embed_dim)
        
        # 5. Attention Block (Section 3.6 placeholder - using Standard Multihead Attention)
        # We will manually pass our computed hk and hv into it.
        self.num_heads = 4
        self.head_dim = embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x):
        # x: [Batch_Size, Sequence_Length] (Token IDs)
        
        # --- Step 1: Token Embedding ---
        he = self.embedding(x)  # [B, T, embed_dim]
        
        # --- Step 2 & 3: Dimension Projection & Convolutions ---
        h = self.project_to_target(he)  # Up-project to target_dim
        
        for conv_block in self.conv_blocks:
            h = conv_block(h)           # Process through time-dependent blocks
            
        # --- Step 4: Project back to create Key vectors (hk) ---
        hk = self.project_to_embed(h)   # [B, T, embed_dim]
        
        # --- Step 5: Value Fusion (hv = sqrt(0.5) * (hk + he)) ---
        # Note: math.sqrt(0.5) is approximately 0.7071
        hv = math.sqrt(0.5) * (hk + he)  # [B, T, embed_dim]
        
        # --- Step 6: Attention Mechanism (Section 3.6 Context Vector calculation) ---
        # hk serves as the Keys (and typically Queries in a self-attention setup)
        # hv serves as the Values
        context_vector = self.compute_attention(queries=hk, keys=hk, values=hv)
        
        return context_vector, hk, hv

    def compute_attention(self, queries, keys, values):
        """
        A simplified implementation of Scaled Dot-Product Attention 
        using the text's designated Keys (hk) and combined Values (hv).
        """
        B, T, C = queries.shape
        
        # Project and split into Multiple Heads: [B, T, H, D_head] -> [B, H, T, D_head]
        q = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = keys.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = values.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention weights using Key vectors (hk)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, T]
        
        # Compute the weighted average over the fused Value vectors (hv)
        context = torch.matmul(attn_weights, v)  # [B, H, T, D_head]
        
        # Concatenate heads back together
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return context

# --- Example Usage / Verification ---
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 50     # e.g., number of unique characters or phonemes
    EMBED_DIM = 256     # Raw token embedding size
    TARGET_DIM = 512    # Core processing hidden size
    
    # Dummy input: Batch of 2 sentences, each with 15 tokens
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, 15))
    
    # Initialize the model
    encoder = TextEncoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, target_dim=TARGET_DIM)
    
    # Run forward pass
    context_out, key_out, value_out = encoder(input)
    
    print(f"Input Shape:          {dummy_input.shape}")
    print(f"Key Vectors (hk):     {key_out.shape}")
    print(f"Value Vectors (hv):   {value_out.shape}")
    print(f"Final Context Vector: {context_out.shape}")
