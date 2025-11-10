import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modeling/"
    )
)
from modeling_fpt2 import FPT2ForSequenceClassification, FPT2LMHeadModel

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def info(text):
    print(f"{bcolors.OKBLUE}{text}{bcolors.ENDC}")

def good(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")

def bad(text):
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", "-m", required=True, default="data/runsimdb", help="Path to the pruned model directory")
    parser.add_argument("--data_path", "-d", default="./data/datasets/imdb/", help="Path to the IMDB dataset directory")
    parser.add_argument("--split", "-s", default="test", help="Dataset split to evaluate on")
    parser.add_argument("--num_examples", "-n", default=None, type=int, help="Number of examples to evaluate (default: all)")
    parser.add_argument("--device", "-D", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", "-b", default=16, type=int)
    parser.add_argument("--out_path", "-o", default=None, help="Path to save predictions as JSON")
    parser.add_argument("--with_embedding_nodes", action="store_true", help="Force with_embedding_nodes=True for custom model")
    return parser.parse_args()

def detect_model_type(model_dir, force_with_embedding_nodes=False):
    config_path = os.path.join(model_dir, "config.json")
    model_class = None
    with_embedding_nodes = False
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        archs = config.get("architectures", [])
        if "FPT2ForSequenceClassification" in archs or "FPT2LMHeadModel" in archs:
            model_class = "fpt2"
            # Optionally, check for custom keys in config for with_embedding_nodes
            with_embedding_nodes = config.get("with_embedding_nodes", False)
        else:
            model_class = "hf"
    else:
        print("[!] config.json not found in model directory. Defaulting to FPT2ForSequenceClassification.")
        model_class = "fpt2"
    if force_with_embedding_nodes:
        with_embedding_nodes = True
    print(f"[i] Model class detected: {model_class} (with_embedding_nodes={with_embedding_nodes})")
    return model_class, with_embedding_nodes

@torch.no_grad()
def main():
    args = parse_args()
    print("Arguments:", args)
    print("Model directory:", args.model_name_or_path)
    print("Data directory:", args.data_path)

    model_class, with_embedding_nodes = detect_model_type(args.model_name_or_path, args.with_embedding_nodes)

    info("[i] Loading model and tokenizer...")
    print("Model class:", model_class)
    print("With embedding nodes:", with_embedding_nodes)
    print("Model config path:", os.path.join(args.model_name_or_path, "config.json"))
    print("Tokenizer: gpt2")
    if model_class == "fpt2":
        from transformers import GPT2Config
        config = GPT2Config.from_pretrained(args.model_name_or_path)
        model = FPT2ForSequenceClassification(config, with_embedding_nodes=with_embedding_nodes)
        weights_bin = os.path.join(args.model_name_or_path, "pytorch_model.bin")
        weights_safe = os.path.join(args.model_name_or_path, "model.safetensors")
        if os.path.exists(weights_bin):
            state_dict = torch.load(weights_bin, map_location=args.device)
        elif os.path.exists(weights_safe):
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(weights_safe, device=args.device)
        else:
            raise FileNotFoundError(f"No model weights found in {args.model_name_or_path}")
        model.load_state_dict(state_dict)
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    info("[i] Loading data...")
    data = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(args.data_path, "train.csv"),
            "test": os.path.join(args.data_path, "test.csv"),
            "validation": os.path.join(args.data_path, "validation.csv"),
        }
    )[args.split]
    print("Dataset columns:", data.column_names)
    print("Number of examples:", len(data))
    if args.num_examples is not None and args.num_examples < len(data):
        data = data.select(range(args.num_examples))

    info(f"[i] Evaluating on {len(data)} examples...")

    correct = 0
    total = 0
    predictions_list = []

    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i+args.batch_size]
        if i == 0:
            print("First batch:", batch)
        texts = batch["text"]
        labels = [1 if s == "positive" else 0 for s in batch["sentiment"]]

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(args.device)
        attention_mask = encodings["attention_mask"].to(args.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        print(f"Batch {i//args.batch_size}: logits shape: {logits.shape}")

        # If logits are [batch, seq_len, vocab], get the last token
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        preds = logits.argmax(dim=-1).cpu().tolist()

        for j, (pred, label) in enumerate(zip(preds, labels)):
            # Print model output for each example in the batch
            print(f"\nExample {i + j + 1}:")
            print("Text:", texts[j])
            print("Logits:", logits[j].cpu().tolist())
            print("Predicted label:", pred)
            print("True label:", label)
            print("Correct:", int(pred == label))
            
            is_correct = int(pred == label)
            correct += is_correct
            total += 1
            predictions_list.append({
                "text": texts[j],
                "true_label": label,
                "pred_label": pred,
                "correct": is_correct
            })

    accuracy = correct / total if total > 0 else 0.0
    info(f"[i] Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Total correct: {correct}, Total: {total}, Accuracy: {accuracy}")

    if args.out_path is not None:
        info(f"[i] Saving predictions to {args.out_path}...")
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(predictions_list, f, indent=2)

if __name__ == "__main__":
    main()