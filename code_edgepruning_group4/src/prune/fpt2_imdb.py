#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
import torch.nn as nn

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/modelling/"
    )
)
from modeling_fpt2 import FPT2LMHeadModel, FPT2ForSequenceClassification

logger = logging.getLogger(__name__)
from transformers.training_args import TrainingArguments
@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(
        default="./data/datasets/imdb/",
        metadata={"help": "The path to the IMDB dataset directory."},
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column containing the review text."}
    )
    label_column: Optional[str] = field(
        default="sentiment",
        metadata={"help": "The name of the column containing the sentiment labels."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging or quicker training, truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging or quicker training, truncate the number of evaluation examples."}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    start_edge_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "Initial edge sparsity of the model."}
    )
    target_edge_sparsity: Optional[float] = field(
        default=0.98,
        metadata={"help": "Target edge sparsity of the model."}
    )
    start_layer_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "Initial layer sparsity of the model."}
    )
    target_layer_sparsity: Optional[float] = field(
        default=0.68,
        metadata={"help": "Target layer sparsity of the model."}
    )
    num_sparsity_warmup_steps: Optional[int] = field(
        default=10,
        metadata={"help": "Number of steps to reach target sparsity."}
    )
    edge_learning_rate: Optional[float] = field(
        default=0.6,
        metadata={"help": "Learning rate for edge parameters."}
    )
    layer_learning_rate: Optional[float] = field(
        default=0.6,
        metadata={"help": "Learning rate for layer parameters."}
    )
    reg_edge_learning_rate: Optional[float] = field(
        default=0.6,
        metadata={"help": "Learning rate for edge regularization."}
    )
    reg_layer_learning_rate: Optional[float] = field(
        default=0.6,
        metadata={"help": "Learning rate for layer regularization."}
    )
    warmup_type: Optional[str] = field(
        default="linear",
        metadata={"help": "Type of warmup for sparsity."}
    )

@dataclass
class ModelArguments:
    initialize_from: str = field(
        default="gpt2",
        metadata={"help": "Model to initialize from."}
    )
    with_embedding_nodes: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to include embedding nodes"}
    )

class IMDBDataCollator:
    def __init__(self, tokenizer, max_length, text_column="review", label_column="sentiment"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    def __call__(self, examples):
        texts = [f"Movie Review: {example[self.text_column]}\nSentiment:" for example in examples]
        labels = [1 if example[self.label_column] == "positive" else 0 for example in examples]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(labels)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # get logits
    # If predictions is 2D, take argmax over last axis
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=-1)
    labels = labels[:, -1] if labels.ndim == 2 else labels

    # Print sample predictions and true labels
    print(f"\n--- Evaluation Metrics ---")
    print(f"Sample Predictions: {predictions[:5].tolist()}")
    print(f"Sample True Labels: {labels[:5].tolist()}")

    accuracy = (predictions == labels).mean()
    print(f"Overall Accuracy: {accuracy.item():.4f}")
    print(f"--------------------------\n")
    return {"accuracy": accuracy}

def freeze_all_except_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def get_optimizers(model, edges_lr, layers_lr, reg_edges_lr, reg_layers_lr, num_training_steps, warmup_steps=0):
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []

    for n, p in model.named_parameters():
        if 'write_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'read_log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda_edge' in n:
            optimizer_2_group.append(p)
        elif 'sparsity_lambda_node' in n:
            optimizer_4_group.append(p)
    
    optimizer = torch.optim.AdamW(
        [
            {'params': optimizer_1_group, 'lr': edges_lr},
            {'params': optimizer_2_group, 'maximize': True, 'lr': reg_edges_lr},
            {'params': optimizer_3_group, 'lr': layers_lr},
            {'params': optimizer_4_group, 'maximize': True, 'lr': reg_layers_lr}
        ],
        lr=edges_lr
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

class FPT2IMDBTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.5)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_layer_sparsity = kwargs.pop('target_layer_sparsity', 0.5)
        self.start_layer_sparsity = kwargs.pop('start_layer_sparsity', 0.0)
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_layer_sparsity_warmup_steps" in kwargs:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_layer_sparsity_warmup_steps')
        else:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', self.num_edge_sparsity_warmup_steps)
        _ = kwargs.pop('num_sparsity_warmup_steps', None)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.gpt2_model = kwargs.pop('gpt2_model', None)
        self.skip_layer_loss_if_higher_sparsity = kwargs.pop('skip_layer_loss_if_higher_sparsity', False)
        
        self.digits = None
        self.device_count = 1
                
        super().__init__(*args, **kwargs)
        
        self.tokenizer = kwargs.pop('tokenizer', None)
        if self.processing_class is not None and hasattr(self.processing_class, "pad_token_id"):
            pad_token_id = (
            self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else self.processing_class.eos_token_id
        )
       
        

    def get_current_edge_target_sparsity(self, global_step):
        if global_step < self.num_edge_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_edge_sparsity + (self.target_edge_sparsity - self.start_edge_sparsity) * 
                    global_step / self.num_edge_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_edge_sparsity) + (math.log(1 - self.target_edge_sparsity) - 
                    math.log(1 - self.start_edge_sparsity)) * global_step / self.num_edge_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_edge_sparsity
        
    def get_current_layer_target_sparsity(self, global_step):
        if global_step < self.num_layer_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_layer_sparsity + (self.target_layer_sparsity - self.start_layer_sparsity) * 
                    global_step / self.num_layer_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_layer_sparsity) + (math.log(1 - self.target_layer_sparsity) - 
                    math.log(1 - self.start_layer_sparsity)) * global_step / self.num_layer_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_layer_sparsity
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get the model outputs
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=self.get_current_layer_target_sparsity(self.state.global_step)
        )
        
        # Calculate the total loss
        logits = outputs["logits"]
        labels = inputs["labels"]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        print(f"Classification Loss: {loss.item()}")
        # # Add regularization losses if they exist
        # if hasattr(outputs, 'edge_loss') and outputs.edge_loss is not None:
        #     loss = loss + outputs.edge_loss
        # if hasattr(outputs, 'node_loss') and outputs.node_loss is not None:
        #     loss = loss + outputs.node_loss
            
        reg_edge_loss = outputs["edge_loss"]
        reg_layer_loss = outputs["node_loss"]
        model_node_sparsity = outputs["model_node_sparsity"]
        target_node_sparsity = outputs["target_node_sparsity"]
        loss = loss+reg_edge_loss+reg_layer_loss
        
        print(f"Main loss: {loss.item()}, Edge loss: {reg_edge_loss.item()}, Layer loss: {reg_layer_loss.item()}")
        
        # Add this to print the number of active edges at each step
        num_active_edges = len(model.transformer.get_edges())
        print(f"Number of active edges: {num_active_edges}")
        
        if return_outputs:
            return loss, outputs
        return loss

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set output directory and sample limits
    training_args.output_dir = "D:/imdb_model3"
    data_args.max_train_samples = 1000
    data_args.max_eval_samples = 1000
    training_args.logging_steps = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    print(f"Output directory: {training_args.output_dir}")
    print(f"Max train samples: {data_args.max_train_samples}")
    print(f"Max eval samples: {data_args.max_eval_samples}")
    print(f"Logging steps: {training_args.logging_steps}")

    # Set seed
    set_seed(training_args.seed)

    # Set training arguments
    training_args.do_train = True
    training_args.do_eval = True
    training_args.num_train_epochs = 3.0
    # For 1000 samples, batch size 8: 1 epoch = 125 steps, 2 epochs = 250 steps
    training_args.max_steps = 250  # 2 epochs
    training_args.per_device_train_batch_size = 8
    training_args.per_device_eval_batch_size = 8
    training_args.logging_steps = 10
    training_args.save_steps = 50
    training_args.output_dir = "D:/imdb_model3"
    training_args.remove_unused_columns = False
    training_args.target_edge_sparsity = data_args.target_edge_sparsity
    training_args.target_node_sparsity = data_args.target_layer_sparsity
    training_args.learning_rate = 1

    # Load dataset
    if os.path.exists(data_args.dataset_path):
        print(f"Loading dataset from {data_args.dataset_path}")
        dataset = load_dataset(
            "csv",
            data_dir=data_args.dataset_path,
            data_files={
                "train": "train.csv",
                "validation": "validation.csv",
                "test": "test.csv"
            }
        )
        print(f"Dataset loaded. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
        # Truncate to max samples if set
        
        if data_args.max_train_samples is not None:
            dataset["train"] = dataset["train"].select(range(min(data_args.max_train_samples, len(dataset["train"]))))
            print(f"Truncated train set to {len(dataset['train'])} samples.")
        if data_args.max_eval_samples is not None:
            dataset["test"] = dataset["test"].select(range(min(data_args.max_eval_samples, len(dataset["test"]))))
            print(f"Truncated test set to {len(dataset['test'])} samples.")
        print(f"Train columns: {dataset['train'].column_names}")
        print(f"Test columns: {dataset['test'].column_names}")
        print(f"Sample train example: {dataset['train'][0]}")
        print(f"Sample test example: {dataset['test'][0]}")
    else:
        raise ValueError(f"Dataset path {data_args.dataset_path} does not exist")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    config = GPT2Config.from_pretrained("gpt2")  # Changed to load directly from gpt2
    model = FPT2ForSequenceClassification(
        config=config,
        num_labels=2,
        with_embedding_nodes=model_args.with_embedding_nodes
    )
    print("Model config:", model.transformer.config)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    # Freeze non-pruning parameters
    print("Freezing non-pruning parameters...")
    freeze_all_except_pruning_params(model)
    print("Parameters frozen.")

    # After model and parameter initialization, before training
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'get_edges'):
        print("Getting initial edges...")
        initial_edges = model.transformer.get_edges()
        print(f"Number of initial edges: {len(initial_edges)}")
        print(f"Sample initial edges: {initial_edges[:10]}")
        initial_edges_path = os.path.join(training_args.output_dir, "initial_edges.json")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(initial_edges_path, "w", encoding="utf-8") as f:
            json.dump(initial_edges, f, indent=2)
        print(f"Initial edges saved to {initial_edges_path}")

    if hasattr(model.transformer, 'get_edge_masks'):
        print("Edge masks (raw):", model.transformer.get_edge_masks())

    # Data collator
    print("Creating data collator...")
    data_collator = IMDBDataCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        text_column=data_args.text_column,
        label_column=data_args.label_column
    )
    print("Data collator created.")

    # Optimizer
    print("Creating optimizers...")
    optimizers = get_optimizers(
        model,
        edges_lr=data_args.edge_learning_rate,
        layers_lr=data_args.layer_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_layers_lr=data_args.reg_layer_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps
    )
    print("Optimizers created.")

    # Initialize trainer
    print("Initializing trainer...")
    trainer = FPT2IMDBTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        skip_layer_loss_if_higher_sparsity=True,
    )
    print("Trainer initialized.")

    # Training
    training_args.do_train = True
    if training_args.do_train:
        print("Starting training...")
        train_result = trainer.train()
        print("Training finished.")
        metrics = train_result.metrics
        print(f"Training completed with metrics: {metrics}")
        
        # Save model
        print("Saving model...")
        trainer.save_model()
        print("Model saved.")
        
        # Save training arguments
        print("Saving training arguments...")
        with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
            f.write(training_args.to_json_string())
        print("Training arguments saved.")
        
        # Save metrics
        print("Logging and saving metrics...")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("Metrics and state saved.")
        
        print(f"Model saved to {training_args.output_dir}")

    # After training, print and save final edges
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'get_edges'):
        print("Getting final remaining edges...")
        final_edges = model.transformer.get_edges()
        print(f"Number of remaining edges: {len(final_edges)}")
        print(f"Sample edges: {final_edges[:10]}")
        edges_path = os.path.join(training_args.output_dir, "final_edges.json")
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(final_edges, f, indent=2)
        print(f"Final edges saved to {edges_path}")

if __name__ == "__main__":
    main()
