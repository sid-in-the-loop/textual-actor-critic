from transformers import AutoTokenizer

# Use local checkpoint tokenizer to avoid HuggingFace network issues
model_path = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer from local checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize(input):
    tokens = tokenizer.encode(input)
    token_length = len(tokens)
    return token_length


def main():
    input = "Hello, world!"
    print(tokenize(input))


if __name__ == "__main__":
    main()



