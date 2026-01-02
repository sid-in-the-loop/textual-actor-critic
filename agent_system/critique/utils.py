from transformers import AutoTokenizer

# Use local checkpoint tokenizer to avoid HuggingFace network issues
model_path = "/data/group_data/cx_group/behavior_priming/checkpoint/qwen3_1.7b/web_qwen_sft_behavior/checkpoint-924"

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



