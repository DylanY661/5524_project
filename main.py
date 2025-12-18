#!/usr/bin/env python3
"""
Evaluation script for advanced fusion models.

Models:
- advanced_mlp (Model 1): Projects to 512-dim CLIP text space
- advanced_joint (Model 2): Projects to shared 1280-dim latent space

Datasets:
- ImageNet-1K (in-distribution)
- ImageNet-R (OOD)
- CIFAR-100 (OOD)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import clip
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_DIM, CLIP_DIM = 768, 512


class MLPFusion(nn.Module):
    """
    Model 1: Concatenate DINO and CLIP embeddings, project to CLIP text space (512-dim).
    """
    def __init__(self, dino_dim=DINO_DIM, clip_dim=CLIP_DIM, hidden_dim=1024, output_dim=512, dropout=0.1, residual_weight=0.5):
        super().__init__()
        
        input_dim = dino_dim + clip_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.residual_alpha = nn.Parameter(torch.tensor(residual_weight))
    
    def forward(self, dino_emb, clip_emb):
        x = torch.cat([dino_emb, clip_emb], dim=-1)
        fused = self.fusion(x)
        alpha = torch.clamp(self.residual_alpha, 0.0, 1.0)
        output = fused + alpha * clip_emb
        return output


class JointFusion(nn.Module):
    """
    Model 2: Joint fusion with separate image and text encoders in shared 1280-dim space.
    """
    def __init__(self, dino_dim=DINO_DIM, clip_dim=CLIP_DIM, image_hidden_dim=768, text_hidden_dim=768, dropout=0.1):
        super().__init__()
        
        self.concat_dim = dino_dim + clip_dim
        
        self.image_encoder = nn.Sequential(
            nn.Linear(self.concat_dim, image_hidden_dim),
            nn.LayerNorm(image_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_hidden_dim, self.concat_dim),
            nn.LayerNorm(self.concat_dim),
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(clip_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_hidden_dim, self.concat_dim),
            nn.LayerNorm(self.concat_dim),
        )
    
    def forward(self, dino_emb, clip_emb):
        x = torch.cat([dino_emb, clip_emb], dim=-1)
        output = self.image_encoder(x)
        return output
    
    def encode_text(self, text_features):
        return self.text_encoder(text_features)

MODEL_CONFIGS = {
    'advanced_1': {
        'checkpoint': 'checkpoints/mlp/best_model.pt',
        'model_class': MLPFusion,
        'model_kwargs': {'hidden_dim': 1024, 'dropout': 0.1, 'residual_weight': 0.5},
    },
    'advanced_2': {
        'checkpoint': 'checkpoints/joint/best_model.pt',
        'model_class': JointFusion,
        'model_kwargs': {'image_hidden_dim': 768, 'text_hidden_dim': 768, 'dropout': 0.1},
    },
}


def load_model(model_name, checkpoint_dir="."):
    """Load a model from checkpoint."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    checkpoint_path = os.path.join(checkpoint_dir, config['checkpoint'])
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading {model_name} from {checkpoint_path}...")
    
    model = config['model_class'](**config['model_kwargs']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    #different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"  Loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_text_embeddings(class_names):
    """Create CLIP text embeddings for class names."""
    print(f"Creating text embeddings for {len(class_names)} classes...")
    
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features.float()


def evaluate_zs(model, clip_embeddings, dino_embeddings, test_labels, text_features, model_name):
    """Zero-shot evaluation for fusion models."""
    model.eval()
    
    clip_emb = clip_embeddings.to(device)
    dino_emb = dino_embeddings.to(device)
    text_features = text_features.to(device)
    
    # For joint model, project text features
    if model_name == 'advanced_2':
        with torch.no_grad():
            eval_text = model.encode_text(text_features)
            eval_text = F.normalize(eval_text, dim=-1)
    else:
        eval_text = text_features
    
    predictions = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(test_labels), batch_size):
            batch_dino = dino_emb[i:i+batch_size]
            batch_clip = clip_emb[i:i+batch_size]
            
            fused = model(batch_dino, batch_clip)
            fused = F.normalize(fused, dim=-1)
            
            similarity = fused @ eval_text.T
            preds = similarity.argmax(dim=-1)
            predictions.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(test_labels.cpu().numpy(), predictions)
    return accuracy

def benchmark_imagenet(model, model_name, embeddings_dir):
    """Benchmark on ImageNet-1K."""
    print("\n" + "="*50)
    print("ImageNet-1K BENCHMARK")
    print("="*50)
    
    # Load embeddings
    clip_data = torch.load(os.path.join(embeddings_dir, "imagenet_clip_embeddings.pt"), weights_only=False)
    dino_data = torch.load(os.path.join(embeddings_dir, "imagenet_dino_embeddings.pt"), weights_only=False)
    
    test_clip = clip_data['test_embeddings'].float()
    test_dino = dino_data['test_embeddings'].float()
    test_labels = clip_data['test_labels']
    
    # Get class names
    dataset = load_dataset("imagenet-1k", split="validation")
    class_names = dataset.features['label'].names
    
    text_features = create_text_embeddings(class_names)
    accuracy = evaluate_zs(model, test_clip, test_dino, test_labels, text_features, model_name)
    print(f"ImageNet-1K Zero-Shot Accuracy: {accuracy*100:.2f}%")
    
    return accuracy


def benchmark_imagenet_r(model, model_name, embeddings_dir):
    """Benchmark on ImageNet-R."""
    print("\n" + "="*50)
    print("ImageNet-R BENCHMARK")
    print("="*50)
    # Load embeddings
    clip_data = torch.load(os.path.join(embeddings_dir, "imagenet_r_clip_embeddings.pt"), weights_only=False)
    dino_data = torch.load(os.path.join(embeddings_dir, "imagenet_r_dino_embeddings.pt"), weights_only=False)
    
    test_clip = clip_data['embeddings'].float()
    test_dino = dino_data['embeddings'].float()
    test_labels = dino_data['labels']
    class_names = clip_data['class_names']
    
    text_features = create_text_embeddings(class_names)
    accuracy = evaluate_zs(model, test_clip, test_dino, test_labels, text_features, model_name)
    print(f"ImageNet-R Zero-Shot Accuracy: {accuracy*100:.2f}%")
    
    return accuracy


def benchmark_cifar100(model, model_name, embeddings_dir):
    """Benchmark on CIFAR-100."""
    print("\n" + "="*50)
    print("CIFAR-100 BENCHMARK")
    print("="*50)
    
    # Load embeddings
    clip_data = torch.load(os.path.join(embeddings_dir, "cifar100_clip_embeddings.pt"), weights_only=False)
    dino_data = torch.load(os.path.join(embeddings_dir, "cifar100_dino_embeddings.pt"), weights_only=False)
    
    test_clip = clip_data['test_embeddings'].float()
    test_dino = dino_data['test_embeddings'].float()
    test_labels = clip_data['test_labels']
    
    # Get class names
    cifar = load_dataset("cifar100", split="test")
    class_names = cifar.features['fine_label'].names
    text_features = create_text_embeddings(class_names)
    accuracy = evaluate_zs(model, test_clip, test_dino, test_labels, text_features, model_name)
    print(f"CIFAR-100 Zero-Shot Accuracy: {accuracy*100:.2f}%")
    
    return accuracy

def print_summary(results, model_name):
    """Print final results summary."""
    print("\n" + "="*60)
    print(f"RESULTS SUMMARY: {model_name}")
    print("="*60)
    
    for dataset, accuracy in results.items():
        print(f"  {dataset}: {accuracy*100:.2f}% (Zero-Shot)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion models on benchmarks")
    parser.add_argument('--model', '-m', type=str, required=True,choices=list(MODEL_CONFIGS.keys()) + ['all'], default='all')
    parser.add_argument('--embeddings_dir', '-e', type=str, default='embeddings')
    parser.add_argument('--checkpoint_dir', '-c', type=str, default='.')
    parser.add_argument('--dataset', '-d', type=str, default='all',choices=['imagenet', 'imagenet-r', 'cifar100', 'all'])
    args = parser.parse_args()
    
    print(f"Device: {device}")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Determine which models to evaluate
    if args.model == 'all':
        model_names = list(MODEL_CONFIGS.keys())
    else:
        model_names = [args.model]
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {model_name}")
        print(f"{'#'*60}")
        
        # Load model
        model = load_model(model_name, args.checkpoint_dir)
        
        results = {}
        
        # Run benchmarks
        if args.dataset in ['imagenet', 'all']:
            results['ImageNet-1K'] = benchmark_imagenet(model, model_name, args.embeddings_dir)
        
        if args.dataset in ['imagenet-r', 'all']:
            try:
                results['ImageNet-R'] = benchmark_imagenet_r(model, model_name, args.embeddings_dir)
            except FileNotFoundError as e:
                print(f"  Skipping ImageNet-R: {e}")
        
        if args.dataset in ['cifar100', 'all']:
            try:
                results['CIFAR-100'] = benchmark_cifar100(model, model_name, args.embeddings_dir)
            except FileNotFoundError as e:
                print(f"  Skipping CIFAR-100: {e}")
        
        all_results[model_name] = results
        print_summary(results, model_name)
    
    # Print comparison table
    if len(model_names) > 1:
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        # Header
        header = f"{'Model':<20}"
        datasets = ['ImageNet-1K', 'ImageNet-R', 'CIFAR-100']
        for ds in datasets:
            header += f" | {ds:>12}"
        print(header)
        print("-" * len(header))
        
        # Rows
        for model_name, results in all_results.items():
            row = f"{model_name:<20}"
            for ds in datasets:
                if ds in results:
                    row += f" | {results[ds]*100:>11.2f}%"
                else:
                    row += f" | {'N/A':>12}"
            print(row)


if __name__ == "__main__":
    main()