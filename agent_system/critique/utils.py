from transformers import AutoTokenizer

# Use HuggingFace Hub Qwen3-1.7B tokenizer
model_name = "Qwen/Qwen3-1.7B"

# Load tokenizer from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(input):
    tokens = tokenizer.encode(input)
    token_length = len(tokens)
    return token_length


def main():
    input = "Hello, world!"
    print(tokenize(input))


if __name__ == "__main__":
    main()



