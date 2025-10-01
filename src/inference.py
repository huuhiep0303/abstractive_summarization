from run_evaluate import load_trained_model, generate_summary
from tokenizer import load_merges, encode

def summarize_text(text: str, model_path='models/summarizer_model.pth'):
    """Summarize a single text"""
    merges = load_merges('bpe_tokenizer/bpe_merges.json')
    model = load_trained_model(model_path, vocab_size=10000)
    
    tokens = encode(text, merges)
    summary_tokens = generate_summary(model, tokens)
    
    # Convert back to text
    return " ".join([f"token_{t}" for t in summary_tokens])