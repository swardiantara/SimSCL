import pandas as pd
import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import argparse

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='embeddings')
    parser.add_argument("--model_name_or_path", default='all-mpnet-base-v2', type=str,
                        help="Path to pre-trained model or shortcut name of Huggingface model. Default: `all-mpnet-base-v2`.")
    parser.add_argument("--dataset", type=str, default='AAPD',
                    help="Dataset to use for fine-tuning the embedding. Default: `AAPD`")
    parser.add_argument("--label_name", type=str, choices=['first', 'second', 'pair'], default='first',
                    help="Label to use for constructing sample pairs. Default: `pair`")
    parser.add_argument("--strategy", type=str, choices=['single', 'multi'], default='single',
                    help="Either using single or multi-stage fine-tuning. Default: `single`")
    parser.add_argument("--stage", type=int, default=1, help="If `strategy`=`multi`, which stage to run?. Default: `1`.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name", type=str, help="The name of the resulted model.")
    parser.add_argument("--source_scenario", type=str, default='first_single_1')
    parser.add_argument("--margin", type=float, default=0.5, help="Hyperparam to push the negative pair at least `$m$` margin apart. Default: `0.5`")
    parser.add_argument("--exclude_duplicate_negative", action='store_true', help="Whether to exclude negative pair of the same sample.")
    
    args = parser.parse_args()

    args.scenario = "_".join([args.label_name, args.strategy, str(args.stage)])
    output_dir = os.path.join(args.output_dir, args.dataset, args.scenario)
    args.model_name = f'{args.dataset}_{args.scenario}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args.output_dir = output_dir

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    return args


def get_dataset_path(args, split='train'):
    if args.dataset == 'AAPD':
        text_path = os.path.join("datasets", args.dataset, f"text_{split}")
        label_path = os.path.join("datasets", args.dataset, f"label_{split}")
    else:
        raise NotImplementedError
    
    return text_path, label_path


def construct_dataset(text_path, label_path, silence=False):
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
        
    train_texts = read_file(text_path)
    train_labels = [label.split() for label in read_file(label_path)]
    
    sents_m, first_m, second_m, pair_m, sents_s, first_s, second_s, pair_s = [], [], [], [], [], [], [], []
    for text, labels in zip(train_texts, train_labels):
        print(f'labels: {labels}')
        firsts, seconds= [], []
        for label in labels:
            print(f'label: {label}')
            first, second = label.split('.')
            pair = f'{first}-{second}'
            sents_s.append(text)
            first_s.append(first)
            second_s.append(second)
            pair_s.append(pair)
            firsts.append(first)
            seconds.append(second)
        
        first_ms, second_ms = "_".join(set(firsts)), "_".join(set(seconds))
        pair_ms = f'{first_ms}-{second_ms}'
        sents_m.append(text)
        first_m.append(first_ms)
        second_m.append(second_ms)
        pair_m.append(pair_ms)

    if not silence:
        print(f"Total examples = {len(sents_m)}")
        print(f"Total quad examples = {len(sents_s)}")
    
    # multilabel dataset for visualization and investigation
    dataframe_multi = pd.DataFrame({
        'text': sents_m,
        'first': first_m,
        'second': second_m,
        'pair': pair_m
    })

    # single label dataset for contrastive fine-tuning
    dataframe_single = pd.DataFrame({
        'text': sents_s,
        'first': first_s,
        'second': second_s,
        'pair': pair_s,
    })

    return dataframe_multi, dataframe_single 


# Create pairs for contrastive learning
def create_pairs(args, dataset: pd.DataFrame) -> list[InputExample]:
    examples = []

    if args.exclude_duplicate_negative:
        for label in dataset[args.label_name].unique():
            cluster_df = dataset[dataset[args.label_name] == label]
            other_df = dataset[dataset[args.label_name] != label]
            for i, row in cluster_df.iterrows():
                for j, other_row in cluster_df.iterrows():
                    # construct positive pairs
                    if i != j and row['text'] != other_row['text']:
                        examples.append(InputExample(texts=[row['text'], other_row['text']], label=1.0))
                for j, other_row in other_df.iterrows():
                    # construct negative pairs
                    if row['text'] != other_row['text']:
                        examples.append(InputExample(texts=[row['text'], other_row['text']], label=0.0))
    else: # include negative pairs containing exactly the same sentence or text
        for label in dataset[args.label_name].unique():
            cluster_df = dataset[dataset[args.label_name] == label]
            other_df = dataset[dataset[args.label_name] != label]
            for i, row in cluster_df.iterrows():
                for j, other_row in cluster_df.iterrows():
                    # construct positive pairs
                    if i != j:
                        examples.append(InputExample(texts=[row['text'], other_row['text']], label=1.0))
                for j, other_row in other_df.iterrows():
                    # construct negative pairs
                    examples.append(InputExample(texts=[row['text'], other_row['text']], label=0.0))
    return examples


def main():
    # initialization
    args = init_args()
    
    # Step 1: Load a pre-trained model
    if args.strategy == 'multi':
        # Load the model from the previous stage
        model_path = os.path.join('embeddings', args.dataset, args.source_scenario)
        model = SentenceTransformer(model_path).to(device)
    else:
        model = SentenceTransformer(args.model_name_or_path).to(device)

    # Load your dataset
    text_path, label_path = get_dataset_path(args, split='train')
    dataframe_multi, dataframe_single = construct_dataset(text_path, label_path)
    dataframe_multi.to_excel(os.path.join(args.output_dir, f'{args.dataset}.xlsx'), index=False)
    dataframe_single.to_excel(os.path.join(args.output_dir, f'single_{args.dataset}.xlsx'), index=False)

    contrastive_samples = create_pairs(args, dataframe_single)
    # contrastive_samples.to_excel(os.path.join(args.output_dir, f'cont_{args.dataset}.xlsx'), index=False)
    # Step 3: Create DataLoader
    train_dataloader = DataLoader(contrastive_samples, shuffle=True, batch_size=args.batch_size)
    # print(train_dataloader)

    # Step 4: Define the contrastive loss
    train_loss = losses.ContrastiveLoss(model=model, margin=args.margin)

    # Optional: Define evaluator for validation
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(contrastive_samples, name=args.model_name)

    # Step 5: Train the model
    warmup_steps = int(len(train_dataloader) * args.num_epochs * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir
    )

    # Save the model
    model.save(args.output_dir, args.model_name)

    return 0


if __name__ == '__main__':
    main()