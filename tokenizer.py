from collections import defaultdict
import re
import json
import unicodedata
from typing import List, Tuple, Dict

def normalize_text(text: str) -> str:
    """Chuẩn hóa text Unicode (NFC) và loại bỏ khoảng trắng thừa."""
    return unicodedata.normalize('NFC', text).strip()

# Biểu thức chính quy để tách các dấu câu phổ biến
_PUNCT_RE = re.compile(r"([.,!?:;()\[\]\"'“”„—–])")

def pre_tokenize(text: str) -> List[str]:
    """Tiền xử lý văn bản: chuẩn hóa, tách dấu câu và chia thành các token."""
    text = normalize_text(text)
    # Thêm khoảng trắng xung quanh dấu câu
    text = _PUNCT_RE.sub(r" \1 ", text)
    # Gộp nhiều khoảng trắng thành một
    text = re.sub(r"\s+", " ", text)
    return text.split(' ')

def bpe(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Áp dụng các quy tắc merge BPE cho một từ."""
    # Thêm ký tự kết thúc từ
    symbols = list(word) + ['</w>']
    merges_set = set(merges)

    while True:
        merged = False
        i = 0
        new_symbols = []
        while i < len(symbols):
            # Kiểm tra nếu cặp token hiện tại có trong danh sách merge
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) in merges_set:
                new_symbols.append(symbols[i] + symbols[i+1])
                i += 2
                merged = True
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
        if not merged:
            break
    
    # Xóa ký tự kết thúc từ nếu nó là token cuối cùng
    if symbols and symbols[-1] == '</w>':
        symbols = symbols[:-1]
        
    return symbols

def encode(text: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Mã hóa toàn bộ văn bản thành một chuỗi các token BPE."""
    tokens = []
    for tok in pre_tokenize(text):
        if tok:
            tokens.extend(bpe(tok, merges))
    return tokens

def load_merges(path: str) -> List[Tuple[str, str]]:
    """Tải các quy tắc merge BPE từ tệp JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        raw_merges = json.load(f)
    return [tuple(pair) for pair in raw_merges]

def save_merges(merges: List[Tuple[str, str]], path: str):
    """Lưu các quy tắc merge BPE vào tệp JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(merges, f, ensure_ascii=False, indent=2)