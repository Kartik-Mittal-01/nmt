import re
import sys

def clean_text(line: str) -> str:
    cleaned = re.sub(r'\s+([?.!,।॥])', r'\1', line)
    cleaned = cleaned.replace("@@ ", "") 
    cleaned = cleaned.replace(" ##", "")
    return cleaned

def detokenize_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(clean_text(line))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_punctuation.py <input.txt> <output.txt>")
    else:
        detokenize_file(sys.argv[1], sys.argv[2])
        print(f"Cleaned {sys.argv[1]} -> saved to {sys.argv[2]}")