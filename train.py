
import os
import sys
import importlib
import argparse
import nltk
import pandas as pd
import numpy as np
import random
import torch

from loader import ArticleLoader
from tokenizer import TokenizerOptimization
from preprocessor import Preprocessor

import wandb
from dotenv import load_dotenv

from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

def train(args):

    # -- Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # -- Checkpoint
    model_checkpoint = args.PLM

    # -- Datasets
    print('\nLoading Article Data')
    article_loader = ArticleLoader('./Data', './theguardians_article_info.csv', args.max_doc_len, args.min_doc_len)
    datasets = article_loader.load_data()
    print(datasets)

    # -- Preprocessor
    print('\nPreprocessing Data')
    preprocessor = Preprocessor()
    datasets.cleanup_cache_files()
    datasets = datasets.map(preprocessor.preprocess4train, load_from_cache_file=True)

    # -- Tokenizer Optimization
    print('\nOptimizing Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    unk_token_data = pd.read_csv('/opt/ml/project/NewsSummarization/Tokenizer/extra_tokens.csv')
    tokenizer_opimizer = TokenizerOptimization(tokenizer, './Tokenizer', unk_token_data)
    tokenizer = tokenizer_opimizer.optimize()
    print('Length of Tokenizer : %d' %len(tokenizer))

    # -- Tokenize
    print('\nTokenizing Data')
    max_input_length = args.max_input_len
    max_target_length = args.max_target_len

    def tokenize_function(examples):
        inputs = ['summarize: ' + doc for doc in examples['document']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    datasets.cleanup_cache_files()
    tokenized_datasets = datasets.map(tokenize_function, batched=True, load_from_cache_file=True)
    print(tokenized_datasets)
    train_data = tokenized_datasets['train']
    val_data = tokenized_datasets['validation']

    # -- Configuration
    config = AutoConfig.from_pretrained(model_checkpoint)
    print(config)

    # -- Model
    print('\nLoading Model')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=config).to(device)
    print('Model Type : {}'.format(type(model)))

    training_args = Seq2SeqTrainingArguments(
        output_dir = args.output_dir,                                   # output directory
        logging_dir = args.logging_dir,                                 # logging directory
        num_train_epochs = args.epochs,                                 # epochs
        save_steps = args.eval_steps,                                   # model saving steps
        eval_steps = args.eval_steps,                                   # evaluation steps
        logging_steps = args.eval_steps,                                # logging steps
        evaluation_strategy = args.evaluation_strategy,                 # evaluation strategy
        per_device_train_batch_size = args.train_batch_size,            # train batch size
        per_device_eval_batch_size = args.eval_batch_size,              # evaluation batch size
        warmup_steps=args.warmup_steps,                                 # warmup steps
        weight_decay=args.weight_decay,                                 # weight decay
        learning_rate = args.learning_rate,                             # learning rate
        gradient_accumulation_steps=args.gradient_accumulation_steps,   # accumulation steps
        fp16=True if args.fp16 == 1 else False,                         # fp 16 flag
        predict_with_generate=True,
        load_best_model_at_end=True
    )

    # -- Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # -- Metric
    metric = load_metric("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    # -- Trainer
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # -- Training
    print('\nTraining')
    trainer.train()

def main(args):
    load_dotenv(dotenv_path=args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = f"epochs:{args.epochs}_batch_size:{args.train_batch_size}_warmup_steps:{args.warmup_steps}_weight_decay:{args.weight_decay}"
    wandb.init(
        entity="sangha0411",
        project="News-summarization", 
        name=wandb_name,
        group='t5-model')

    wandb.config.update(args)
    train(args)
    wandb.finish()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -- Directory
    parser.add_argument('--output_dir', default='./results', help='model save at {SM_SAVE_DIR}/{name}')
    parser.add_argument('--logging_dir', default='./logs', help='logging save at {SM_SAVE_DIR}/{name}')

    # -- Model
    parser.add_argument('--PLM', type=str, default='t5-base', help='model type (default: t5-base)')

    # -- Training
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train (default: 3)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate (default: 5e-5)')
    parser.add_argument('--train_batch_size', type=int, default=16, help='train batch size (default: 16)')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='eval batch size (default: 16)')
    parser.add_argument('--eval_steps', type=int, default=2000, help='evaluation steps (default : 2000)')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='number of warmup steps for learning rate scheduler (default: 4000)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='streng1th of weight decay (default: 1e-2)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps of training (default: 1)')
    parser.add_argument('--fp16', type=int, default=1, help='using fp16 (default: 1)')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='evaluation strategy to adopt during training, steps or epoch (default: steps)')

    # -- Data
    parser.add_argument('--max_input_len', type=int, default=768, help='max length of tokenized document (default: 768)')
    parser.add_argument('--max_target_len', type=int, default=128, help='max length of tokenized summary (default: 128)')
    parser.add_argument('--max_doc_len', type=int, default=24000, help='max length of document (default: 24000)')
    parser.add_argument('--min_doc_len', type=int, default=1000, help='max length of document (default: 1000)')

    # -- Seed
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    # -- Wandb
    parser.add_argument('--dotenv_path', default='/opt/ml/wandb.env', help='input your dotenv path')

    args = parser.parse_args()

    seed_everything(args.seed)   
    main(args)

