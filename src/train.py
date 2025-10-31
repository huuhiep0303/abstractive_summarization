import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.transformer_summarizer import TransformerSummarizer
from src.data_processor import create_data_loaders
from utils.metrics import rouge
from tokenizer import load_merges
from tqdm import tqdm

def create_src_tgt_mask(src_len, tgt_len, device):
    # Source mask (padding mask)
    src_mask = torch.zeros(src_len, src_len).to(device)
    
    # Target mask (causal mask + padding mask)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1).to(device)
    
    return src_mask, tgt_mask

def train_model():
    print("Starting Vietnamese Abstractive Summarization Training...")
    print("=" * 60)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BPE merges
    print("Loading BPE tokenizer...")
    try:
        merges = load_merges('bpe_tokenizer/bpe_merges.json')
        print(f"Loaded {len(merges)} BPE merges")
    except Exception as e:
        print(f"Error loading BPE merges: {e}")
        return
    
    print("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader, vocab_size = create_data_loaders(
            'data/raw/train.json',
            'data/raw/validation.json', 
            'data/raw/test.json',
            merges,
            batch_size=4  # Reduced batch size for memory
        )
        print(f"Data loaders created. Vocab size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    print("Initializing model...")
    model = TransformerSummarizer(
        vocab_size=vocab_size,
        d_model=256,  # Reduced for memory
        nhead=8,
        num_encoder_layers=3,  # Reduced for faster training
        num_decoder_layers=3,
        max_seq_length=1024 # Tăng giá trị này để xử lý các chuỗi dài hơn
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Training loop
    num_epochs = 5  # Reduced for testing
    best_val_loss = float('inf')

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            try:
                optimizer.zero_grad()
                
                # Move data to device
                src = batch['articles'].to(device)
                tgt = batch['summaries'].to(device)
                
                tgt_input = tgt[:, :-1]  # Remove last token
                tgt_output = tgt[:, 1:]  # Remove first token
                
                # Forward pass
                output = model(src, tgt_input)
                
                # Calculate loss
                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_train_loss/num_batches:.4f}'
                })
                
                # Break early for testing
                if batch_idx >= 100:  # Only process first 100 batches
                    break
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = total_train_loss / max(num_batches, 1)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        if epoch % 1 == 0:  # Validate every epoch
            val_loss = validate_model(model, val_loader, criterion, device, vocab_size)
            print(f"Validation Loss: {val_loss:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best model...")
                Path('models').mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'vocab_size': vocab_size
                }, 'models/best_model.pth')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

def validate_model(model, val_loader, criterion, device, vocab_size, max_batches=50):
    """Validate the model"""
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        for batch_idx, batch in enumerate(val_pbar):
            try:
                # Move data to device
                src = batch['articles'].to(device)
                tgt = batch['summaries'].to(device)
                
                # Create input and target for decoder
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward pass
                output = model(src, tgt_input)
                
                # Calculate loss
                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                
                total_val_loss += loss.item()
                num_batches += 1
                
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                
                # Break early for testing
                if batch_idx >= max_batches:
                    break
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return total_val_loss / max(num_batches, 1)

if __name__ == "__main__":
    train_model()