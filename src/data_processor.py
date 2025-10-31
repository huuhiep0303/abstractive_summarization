import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import tokenizer functions
try:
    from tokenizer import encode, load_merges
except ImportError:
    print("Warning: Could not import tokenizer functions. Make sure tokenizer.py exists.")

class SummarizationDataset(Dataset):
    def __init__(self, data_path: str, merges: List[Tuple[str, str]], 
                 max_article_length: int = 512, max_summary_length: int = 128):
        self.data = self.load_data(data_path)
        self.merges = merges
        self.max_article_length = max_article_length
        self.max_summary_length = max_summary_length
        
        # Build vocabulary from merges
        self.vocab = self.build_vocab_from_merges(merges)
        
    def load_data(self, data_path: str) -> List[Dict]:
        # Load JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def build_vocab_from_merges(self, merges: List[Tuple[str, str]]) -> Dict[str, int]:
        # Build vocabulary mapping from BPE merges
        vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        idx = 4
        
        # Add all tokens from merges
        for merge in merges:
            for token in merge:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        
        return vocab
    
    def tokenize_and_encode(self, text: str) -> List[int]:
        # Tokenize text and convert to IDs
        try:
            tokens = encode(text, self.merges)
            # Convert tokens to IDs
            token_ids = []
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.vocab['<unk>'])
            return token_ids
        except:
            words = text.split()[:self.max_article_length]
            return [self.vocab.get(word, self.vocab['<unk>']) for word in words]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        article = item['article']
        summary = item['abstract']
        
        # Tokenize and encode
        article_ids = self.tokenize_and_encode(article)[:self.max_article_length]
        summary_ids = self.tokenize_and_encode(summary)[:self.max_summary_length]
        
        # Add special tokens
        article_ids = [self.vocab['<bos>']] + article_ids + [self.vocab['<eos>']]
        summary_ids = [self.vocab['<bos>']] + summary_ids + [self.vocab['<eos>']]
        
        return {
            'article_ids': torch.tensor(article_ids, dtype=torch.long),
            'summary_ids': torch.tensor(summary_ids, dtype=torch.long),
            'article_text': article,
            'summary_text': summary
        }

def collate_fn(batch):
    max_article_len = max([len(item['article_ids']) for item in batch])
    max_summary_len = max([len(item['summary_ids']) for item in batch])
    
    articles = torch.zeros(len(batch), max_article_len, dtype=torch.long)
    summaries = torch.zeros(len(batch), max_summary_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        article_len = len(item['article_ids'])
        summary_len = len(item['summary_ids'])
        
        articles[i, :article_len] = item['article_ids']
        summaries[i, :summary_len] = item['summary_ids']
    
    return {
        'articles': articles,
        'summaries': summaries,
        'article_texts': [item['article_text'] for item in batch],
        'summary_texts': [item['summary_text'] for item in batch]
    }

def create_data_loaders(train_path: str, val_path: str, test_path: str, 
                       merges: List[Tuple[str, str]], batch_size: int = 8):
    """Create data loaders for training, validation, and testing"""
    
    train_dataset = SummarizationDataset(train_path, merges)
    val_dataset = SummarizationDataset(val_path, merges)
    test_dataset = SummarizationDataset(test_path, merges)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Return vocab_size as well
    vocab_size = len(train_dataset.vocab)
    
    return train_loader, val_loader, test_loader, vocab_size