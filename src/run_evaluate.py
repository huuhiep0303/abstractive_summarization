import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from models.transformer_summarizer import TransformerSummarizer
from utils.metrics import rouge, accuracy, macro_f1, map_name_to_metric_function
from src.data_processor import SummarizationDataset
from tokenizer import load_merges, encode

def load_trained_model(model_path: str, vocab_size: int):
    """Load trained model from checkpoint"""
    model = TransformerSummarizer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def generate_summary(model, article_tokens, max_length=128):
    """Generate summary from article tokens using the trained model"""
    model.eval()
    with torch.no_grad():
        src = torch.tensor([article_tokens])  # Add batch dimension
        # Initialize decoder input with start token
        tgt = torch.tensor([[0]]) 
        
        for _ in range(max_length):
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            # Stop if end token is generated
            if next_token.item() == 1:  # Assuming 1 is end token
                break
                
        return tgt[0].tolist()  # Remove batch dimension

def evaluate_model(model_path='models/summarizer_model.pth', 
                  test_data_path='data/raw/test.json',
                  merges_path='bpe_tokenizer/bpe_merges.json',
                  max_samples=None):
    """Evaluate the trained model on test data"""
    
    print("Starting Model Evaluation...")
    print("=" * 50)
    
    # Load BPE merges
    print("Loading BPE tokenizer...")
    merges = load_merges(merges_path)
    
    # Estimate vocab size (you may need to adjust this)
    vocab_size = len(set([token for merge in merges for token in merge])) + 1000
    
    print("Loading trained model...")
    model = load_trained_model(model_path, vocab_size)
    print("Loading test dataset...")
    test_dataset = SummarizationDataset(test_data_path, merges)
    
    if max_samples:
        test_size = min(max_samples, len(test_dataset))
    else:
        test_size = len(test_dataset)
    print(f"Evaluating on {test_size} samples...")

    predictions = []
    references = []
    
    # Generate predictions
    for i in range(test_size):
        if i % 50 == 0:
            print(f"Processing sample {i}/{test_size}...")
            
        item = test_dataset[i]
        article_tokens = item['article_tokens']
        reference_summary = item['summary_text']
        
        # Generate summary
        predicted_tokens = generate_summary(model, article_tokens)
        
        # Convert tokens back to text (implement based on your tokenizer)
        predicted_summary = tokens_to_text(predicted_tokens, merges)
        
        predictions.append(predicted_summary)
        references.append(reference_summary)
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    
    # ROUGE scores
    rouge_scores = rouge(references, predictions)
    print("\nROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"   {metric}: {score:.4f}")
    
    # Additional metrics (if applicable)
    try:
        # For classification tasks (you may need to adapt this)
        acc_score = accuracy(references, predictions)
        f1_score = macro_f1(references, predictions)

        print(f"\nAdditional Metrics:")
        print(f"   Accuracy: {acc_score['accuracy']:.2f}%")
        print(f"   Macro F1: {f1_score['f1_macro']:.2f}%")
    except:
        print("\nAdditional metrics not applicable for text generation")
    
    # Save results
    results = {
        'rouge_scores': rouge_scores,
        'num_samples': test_size,
        'predictions': predictions[:10],  # Save first 10 for inspection
        'references': references[:10]
    }
    
    results_path = 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {results_path}")

    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 50)
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Reference: {references[i][:100]}...")
        print(f"Predicted: {predictions[i][:100]}...")
    
    return rouge_scores

def tokens_to_text(tokens: List[int], merges: List[Tuple[str, str]]) -> str:
    return " ".join([f"token_{t}" for t in tokens])

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_model(
        model_path='models/summarizer_model.pth',
        test_data_path='data/raw/test.json',
        max_samples=100  # Set to None for full evaluation
    )
    
    print("\nEvaluation completed!")