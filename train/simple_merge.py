#!/usr/bin/env python3
"""
Simple model merging script: merged_model = instruct + alpha * (fine_tuned - base)
Supports both full models and LoRA adapters.

Usage: 
  Full model merge:
    python simple_merge.py --base BASE_MODEL --instruct INSTRUCT_MODEL --fine_tuned FINE_TUNED_MODEL --alpha ALPHA [--output OUTPUT_PATH] [--gpu GPU_ID]
  
  LoRA merge:
    python simple_merge.py --base BASE_MODEL --instruct INSTRUCT_MODEL --lora_path LORA_PATH --alpha ALPHA [--output OUTPUT_PATH] [--gpu GPU_ID]

  General merge:
    python simple_merge.py --A MODEL_A --B MODEL_B --C MODEL_C --D MODEL_D
  
  General merge with LoRA adapter (D-C delta):
    python simple_merge.py --A MODEL_A --B MODEL_B --lora MODEL_D_C_adapter [--beta BETA] [--output OUTPUT_PATH] [--gpu GPU_ID]




Examples:
  Full model merge:
    CUDA_VISIBLE_DEVICES=3 python simple_merge.py \
        --base google/gemma-3-4b-pt \
        --instruct google/gemma-3-4b-it \
        --fine_tuned /path/to/full_model/ \
        --alpha 0.5 \
        --output ./outputs/merged_models/full_merge
  
  LoRA merge (complex formula):
    CUDA_VISIBLE_DEVICES=3 python simple_merge.py \
        --base google/gemma-3-4b-pt \
        --instruct google/gemma-3-4b-it \
        --lora_path /path/to/lora_checkpoint/ \
        --alpha 0.5 \
        --output ./outputs/merged_models/lora_merge
  
  LoRA simple merge (adapter only):
    CUDA_VISIBLE_DEVICES=3 python simple_merge.py \
        --lora_path /path/to/lora_checkpoint/ \
        --lora_base google/gemma-3-4b-pt \
        --output ./outputs/merged_models/lora_simple_merge

  General merge (4 models):
    CUDA_VISIBLE_DEVICES=3 python simple_merge.py \
        --A google/gemma-3-4b-pt \
        --B google/gemma-3-4b-it \
        --C /path/to/model_c/ \
        --D /path/to/model_d/ \
        --beta 0.7 \
        --output ./outputs/merged_models/general_merge
  
  General merge with LoRA (3 models + adapter):
    CUDA_VISIBLE_DEVICES=3 python simple_merge.py \
        --A google/gemma-3-4b-pt \
        --B google/gemma-3-4b-it \
        --lora /path/to/d_c_adapter/ \
        --beta 0.7 \
        --output ./outputs/merged_models/general_lora_merge
"""

import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

try:
    from peft import PeftModel, get_peft_model_state_dict
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description='Simple model merging with custom formula')
    
    # LoRA simple merge arguments (mutually exclusive with complex merge)
    parser.add_argument('--lora_path', type=str,
                       help='LoRA adapter path (e.g., /path/to/checkpoint-1256/)')
    parser.add_argument('--lora_base', type=str,
                       help='Base model for LoRA simple merge (can be base or instruct model)')
    
    # Complex merge arguments (base + instruct + fine_tuned/lora)
    parser.add_argument('--base', type=str,
                       help='Base model path or HuggingFace identifier (e.g., google/gemma-3-4b-pt)')
    parser.add_argument('--instruct', type=str,
                       help='Instruct model path or HuggingFace identifier (e.g., google/gemma-3-4b-it)')
    parser.add_argument('--alpha', type=float,
                       help='Merge weight (0.0 = only instruct, 1.0 = only fine_tuned delta)')
    
    # Model type for complex merge
    parser.add_argument('--fine_tuned', type=str,
                       help='Fine-tuned full model path (e.g., /path/to/cpt-model/)')
    
    # General merge arguments (4-model merge or 3-model + LoRA)
    parser.add_argument('--A', type=str,
                       help='Model A for general merge: C + beta*(B-A) + (1-beta)*(D-C) OR A + beta*(B-A) + (1-beta)*lora_adapter')
    parser.add_argument('--B', type=str,
                       help='Model B for general merge: C + beta*(B-A) + (1-beta)*(D-C) OR A + beta*(B-A) + (1-beta)*lora_adapter')
    parser.add_argument('--C', type=str,
                       help='Model C for general merge: C + beta*(B-A) + (1-beta)*(D-C)')
    parser.add_argument('--D', type=str,
                       help='Model D for general merge: C + beta*(B-A) + (1-beta)*(D-C)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Beta weight for general merge (default: 0.5)')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='./outputs/merged_models/simple_merge',
                       help='Output directory for merged model (default: ./outputs/merged_models/simple_merge)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'bfloat16', 'float16'],
                       help='Model precision (default: float32)')
    
    args = parser.parse_args()
    
    # Validation logic
    if args.lora_path and args.lora_base:
        # LoRA simple merge mode
        return args
    elif args.base and args.instruct and args.alpha is not None and args.lora_path:
        # LoRA complex merge mode
        return args
    elif args.base and args.instruct and args.alpha is not None and args.fine_tuned:
        # Full model merge mode
        return args
    elif args.A and args.B and args.C and args.D:
        # General merge mode (4 models)
        return args
    elif args.A and args.B and args.lora_path and not args.C and not args.D:
        # General merge with LoRA mode (3 models + LoRA adapter)
        return args
    else:
        parser.error("""
Invalid argument combination. Choose one of:

1. LoRA Simple Merge:
   --lora_path LORA_PATH --lora_base BASE_MODEL [--output OUTPUT]

2. LoRA Complex Merge:
   --base BASE --instruct INSTRUCT --lora_path LORA_PATH --alpha ALPHA [--output OUTPUT]

3. Full Model Merge:
   --base BASE --instruct INSTRUCT --fine_tuned FINE_TUNED --alpha ALPHA [--output OUTPUT]

4. General Merge (4 models):
   --A MODEL_A --B MODEL_B --C MODEL_C --D MODEL_D [--beta BETA] [--output OUTPUT]

5. General Merge with LoRA (3 models + adapter):
   --A MODEL_A --B MODEL_B --lora LORA_PATH [--beta BETA] [--output OUTPUT]
        """)
    
    return args

def simple_lora_merge(args, device, dtype):
    """
    Simple LoRA merge: base_model + lora_adapter
    
    Args:
        args: Parsed arguments
        device: Device to use
        dtype: Model precision
    
    Returns:
        Merged model and tokenizer
    """
    print("Loading models for simple LoRA merge...")
    
    # Load tokenizer from base model
    print(f"Loading tokenizer from base model: {args.lora_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_base, trust_remote_code=True)
    
    # Load base model
    print(f"Loading base model: {args.lora_base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.lora_base,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Merge LoRA with base model
    merged_model = merge_lora_with_base(base_model, args.lora_path, device, dtype)
    
    print("✅ Simple LoRA merge completed")
    return merged_model, tokenizer

def merge_lora_with_base(base_model, lora_path, device, dtype):
    """
    Merge LoRA adapter with base model to create fine-tuned model.
    
    Args:
        base_model: Loaded base model
        lora_path: Path to LoRA adapter
        device: Device to use
        dtype: Model precision
    
    Returns:
        Model with LoRA weights merged
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required for LoRA merging. Install with: pip install peft")
    
    print(f"Loading LoRA adapter from: {lora_path}")
    
    # Load LoRA model
    lora_model = PeftModel.from_pretrained(base_model, lora_path, device_map={"": device})
    
    print("Merging LoRA weights with base model...")
    # Merge LoRA weights into base model
    merged_model = lora_model.merge_and_unload()
    
    return merged_model

def merge_full_models(args, device, dtype):
    """
    Merge full models using the formula: merged = instruct + alpha * (fine_tuned - base)
    
    Args:
        args: Parsed arguments
        device: Device to use  
        dtype: Model precision
    
    Returns:
        Merged model and tokenizer
    """
    print("Loading models for full model merge...")
    
    # Load tokenizer from instruct model
    print("Loading tokenizer from instruct model...")
    tokenizer = AutoTokenizer.from_pretrained(args.instruct, trust_remote_code=True)
    
    # Load models with specified precision
    print("Loading instruct model...")
    instruct_model = AutoModelForCausalLM.from_pretrained(
        args.instruct,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    print("Loading fine-tuned model...")
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        args.fine_tuned,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    print(f"\nPerforming merge with alpha={args.alpha}...")
    print("Formula: merged = instruct + alpha * (fine_tuned - base)")
    
    # Get state dictionaries
    instruct_state = instruct_model.state_dict()
    fine_tuned_state = fine_tuned_model.state_dict()
    base_state = base_model.state_dict()
    
    # Perform the merge: merged = instruct + alpha * (fine_tuned - base)
    merged_state = {}
    
    for key in instruct_state.keys():
        if key in fine_tuned_state and key in base_state:
            # Calculate delta: fine_tuned - base
            delta = fine_tuned_state[key] - base_state[key]
            
            # Apply merge formula
            merged_state[key] = instruct_state[key] + args.alpha * delta
            
            print(f"Merged: {key}")
        else:
            # If parameter doesn't exist in all models, use instruct model
            merged_state[key] = instruct_state[key]
            print(f"Copied from instruct: {key}")
    
    # Load the merged weights into the instruct model
    print("\nLoading merged weights...")
    instruct_model.load_state_dict(merged_state)
    
    return instruct_model, tokenizer

def merge_lora_models(args, device, dtype):
    """
    Merge models using LoRA adapter: merged = instruct + alpha * (lora_merged_base - base)
    
    Args:
        args: Parsed arguments
        device: Device to use
        dtype: Model precision
    
    Returns:
        Merged model and tokenizer
    """
    print("Loading models for LoRA merge...")
    
    # Load tokenizer from instruct model
    print("Loading tokenizer from instruct model...")
    tokenizer = AutoTokenizer.from_pretrained(args.instruct, trust_remote_code=True)
    
    # Load models
    print("Loading instruct model...")
    instruct_model = AutoModelForCausalLM.from_pretrained(
        args.instruct,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Create a copy of base model for LoRA merging
    print("Creating base model copy for LoRA merge...")
    base_model_for_lora = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Merge LoRA with base model
    lora_merged_model = merge_lora_with_base(base_model_for_lora, args.lora_path, device, dtype)
    
    print(f"\nPerforming merge with alpha={args.alpha}...")
    print("Formula: merged = instruct + alpha * (lora_merged_base - base)")
    
    # Get state dictionaries
    instruct_state = instruct_model.state_dict()
    lora_merged_state = lora_merged_model.state_dict()
    base_state = base_model.state_dict()
    
    # Perform the merge: merged = instruct + alpha * (lora_merged_base - base)
    merged_state = {}
    
    for key in instruct_state.keys():
        if key in lora_merged_state and key in base_state:
            # Calculate delta: lora_merged_base - base
            delta = lora_merged_state[key] - base_state[key]
            
            # Apply merge formula
            merged_state[key] = instruct_state[key] + args.alpha * delta
            
            print(f"Merged: {key}")
        else:
            # If parameter doesn't exist in all models, use instruct model
            merged_state[key] = instruct_state[key]
            print(f"Copied from instruct: {key}")
    
    # Load the merged weights into the instruct model
    print("\nLoading merged weights...")
    instruct_model.load_state_dict(merged_state)
    
    return instruct_model, tokenizer

def merge_general_lora_models(args, device, dtype):
    """
    General merge using 2 models + LoRA adapter: merged = A + beta * (B - A) + (1 - beta) * lora_adapter
    Ultra-memory-optimized version that computes deltas on-demand without storing them.
    
    Args:
        args: Parsed arguments with A, B models, lora_path, and beta
        device: Device to use
        dtype: Model precision
    
    Returns:
        Merged model and tokenizer
    """
    print("Loading models for general LoRA merge...")
    
    # Load tokenizer from model A (base for the formula)
    print("Loading tokenizer from model A...")
    tokenizer = AutoTokenizer.from_pretrained(args.A, trust_remote_code=True)
    
    # Load model A first (this will be our final model)
    print("Loading model A...")
    model_A = AutoModelForCausalLM.from_pretrained(
        args.A,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    print("Loading model B...")
    model_B = AutoModelForCausalLM.from_pretrained(
        args.B,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Phase 1: Apply beta * (B - A) to model A
    print("Phase 1: Applying beta * (B - A) to model A...")
    print("Formula: A = A + beta * (B - A)")
    
    # Get parameter iterators
    params_B = dict(model_B.named_parameters())
    
    # Store original A parameters for LoRA computation
    print("Storing original model A parameters for LoRA computation...")
    original_A_params = {}
    for name, param in model_A.named_parameters():
        original_A_params[name] = param.detach().clone()
    
    # Update model A parameters with beta * (B - A) on-demand
    for name, param_A in model_A.named_parameters():
        if name in params_B:
            param_B = params_B[name]
            
            # Compute and apply beta * (B - A) directly
            with torch.no_grad():
                delta_BA = param_B.data - param_A.data
                param_A.data = param_A.data + args.beta * delta_BA
                del delta_BA  # Immediately clean up temporary tensor
            
            print(f"Applied beta*(B-A): {name}")
        else:
            print(f"Kept original from A: {name}")
    
    # Release model B to free memory
    print("🗑️  Releasing model B from memory...")
    del model_B, params_B
    torch.cuda.empty_cache()
    
    # Phase 2: Load and apply LoRA adapter
    print("Phase 2: Loading LoRA adapter and computing delta...")
    
    # Load LoRA adapter and merge with original model A to get the adapter weights
    print("Loading original model A copy for LoRA merge...")
    model_A_original = AutoModelForCausalLM.from_pretrained(
        args.A,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    lora_merged_A = merge_lora_with_base(model_A_original, args.lora_path, device, dtype)
    
    # Apply (1 - beta) * (lora_merged_A - A_original) to our modified A
    print("Phase 2: Applying (1 - beta) * LoRA delta to model A...")
    print("Formula: A = A + (1 - beta) * (lora_merged_A - A_original)")
    
    # Get parameter iterator for LoRA merged model
    params_lora = dict(lora_merged_A.named_parameters())
    
    for name, param_A in model_A.named_parameters():
        if name in params_lora and name in original_A_params:
            param_lora = params_lora[name]
            param_A_orig = original_A_params[name]
            
            # Compute and apply (1 - beta) * (lora_merged_A - A_original) directly
            with torch.no_grad():
                delta_lora = param_lora.data - param_A_orig.data
                param_A.data = param_A.data + (1 - args.beta) * delta_lora
                del delta_lora  # Immediately clean up temporary tensor
            
            print(f"Applied (1-beta)*LoRA: {name}")
        else:
            print(f"Kept modified A: {name}")
    
    # Release LoRA models and original parameters
    print("🗑️  Releasing LoRA models and original parameters from memory...")
    del model_A_original, lora_merged_A, params_lora, original_A_params
    torch.cuda.empty_cache()
    
    print(f"\n✅ General LoRA merge completed with beta={args.beta}")
    print("Formula applied: merged = A + beta * (B - A) + (1 - beta) * (lora_merged_A - A)")
    print("Where lora_merged_A - A represents the LoRA adapter delta")
    print("✅ Memory cleanup completed")
    return model_A, tokenizer

def merge_general_models(args, device, dtype):
    """
    General merge using 4 models: merged = C + beta * (B - A) + (1 - beta) * (D - C)
    Ultra-memory-optimized version that computes deltas on-demand without storing them.
    
    Args:
        args: Parsed arguments with A, B, C, D models and beta
        device: Device to use
        dtype: Model precision
    
    Returns:
        Merged model and tokenizer
    """
    print("Loading models for general merge...")
    
    # Load tokenizer from model C (base for the formula)
    print("Loading tokenizer from model C...")
    tokenizer = AutoTokenizer.from_pretrained(args.C, trust_remote_code=True)
    
    # Load model C first (this will be our final model)
    print("Loading model C...")
    model_C = AutoModelForCausalLM.from_pretrained(
        args.C,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    print("Loading model A...")
    model_A = AutoModelForCausalLM.from_pretrained(
        args.A,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    print("Loading model B...")
    model_B = AutoModelForCausalLM.from_pretrained(
        args.B,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Phase 1: Apply beta * (B - A) to model C
    print("Phase 1: Applying beta * (B - A) to model C...")
    print("Formula: C = C + beta * (B - A)")
    
    # Get parameter iterators
    params_A = dict(model_A.named_parameters())
    params_B = dict(model_B.named_parameters())
    
    # Update model C parameters with beta * (B - A) on-demand
    for name, param_C in model_C.named_parameters():
        if name in params_A and name in params_B:
            param_A = params_A[name]
            param_B = params_B[name]
            
            # Compute and apply beta * (B - A) directly
            with torch.no_grad():
                delta_BA = param_B.data - param_A.data
                param_C.data = param_C.data + args.beta * delta_BA
                del delta_BA  # Immediately clean up temporary tensor
            
            print(f"Applied beta*(B-A): {name}")
        else:
            print(f"Kept original from C: {name}")
    
    # Release models A and B to free memory
    print("🗑️  Releasing model A and B from memory...")
    del model_A, model_B, params_A, params_B
    torch.cuda.empty_cache()  # Clear GPU cache
    
    # Load model D
    print("Loading model D...")
    model_D = AutoModelForCausalLM.from_pretrained(
        args.D,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Phase 2: Apply (1 - beta) * (D - C_original) to model C
    # Note: We need to be careful here because C has already been modified
    # So we'll compute D - C_current and adjust for the beta*(B-A) we already added
    print("Phase 2: Applying (1 - beta) * (D - C) to model C...")
    print("Formula: C = C + (1 - beta) * (D - C_original)")
    
    # Get parameter iterator for D
    params_D = dict(model_D.named_parameters())
    
    # We need to subtract the beta*(B-A) we added, then add (1-beta)*(D-C_original)
    # This is equivalent to: C_new = C_original + beta*(B-A) + (1-beta)*(D-C_original)
    # Since we already added beta*(B-A), we need to add (1-beta)*(D-C_current) + beta*(B-A)
    # But that's complex. Let's use a different approach:
    
    # Reload model C original to get clean reference
    print("Loading original model C for reference...")
    model_C_original = AutoModelForCausalLM.from_pretrained(
        args.C,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    params_C_original = dict(model_C_original.named_parameters())
    
    # Now apply (1 - beta) * (D - C_original) to our modified C
    for name, param_C in model_C.named_parameters():
        if name in params_D and name in params_C_original:
            param_D = params_D[name]
            param_C_orig = params_C_original[name]
            
            # Compute and apply (1 - beta) * (D - C_original) directly
            with torch.no_grad():
                delta_DC = param_D.data - param_C_orig.data
                param_C.data = param_C.data + (1 - args.beta) * delta_DC
                del delta_DC  # Immediately clean up temporary tensor
            
            print(f"Applied (1-beta)*(D-C): {name}")
        else:
            print(f"Kept modified C: {name}")
    
    # Release model D and original C
    print("🗑️  Releasing model D and original C from memory...")
    del model_D, model_C_original, params_D, params_C_original
    torch.cuda.empty_cache()  # Clear GPU cache
    
    print(f"\n✅ General merge completed with beta={args.beta}")
    print("Formula applied: merged = C + beta * (B - A) + (1 - beta) * (D - C)")
    print("✅ Memory cleanup completed")
    return model_C, tokenizer

def main():
    args = parse_args()
    
    # Check if LoRA merging is requested but PEFT is not available
    if args.lora_path and not PEFT_AVAILABLE:
        print("ERROR: PEFT library is required for LoRA merging.")
        print("Install with: pip install peft")
        exit(1)
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus} (0 to {num_gpus-1})")
    
    if args.gpu >= num_gpus:
        print(f"ERROR: GPU {args.gpu} not available. Available GPUs: 0 to {num_gpus-1}")
        print("Please use --gpu with a valid GPU ID")
        exit(1)
    
    # Set device
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")
    
    # Set dtype
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    print(f"Using dtype: {dtype}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine merge type and print configuration
    if args.lora_path and args.lora_base:
        # Simple LoRA merge mode
        merge_type = "Simple LoRA Merge"
        print("="*60)
        print(f"MERGE CONFIGURATION - {merge_type}")
        print("="*60)
        print(f"Base model:       {args.lora_base}")
        print(f"LoRA adapter:     {args.lora_path}")
        print(f"Formula:          merged = base + lora_adapter")
        print(f"Output:           {args.output}")
        print("="*60)
    elif args.lora_path:
        # Complex LoRA merge mode
        merge_type = "Complex LoRA Merge"
        print("="*60)
        print(f"MERGE CONFIGURATION - {merge_type}")
        print("="*60)
        print(f"Base model:       {args.base}")
        print(f"Instruct model:   {args.instruct}")
        print(f"LoRA adapter:     {args.lora_path}")
        print(f"Formula:          merged = instruct + {args.alpha} * (lora_merged_base - base)")
        print(f"Alpha:            {args.alpha}")
        print(f"Output:           {args.output}")
        print("="*60)
    elif args.A and args.B and args.C and args.D:
        # General merge mode (4 models)
        merge_type = "General Merge (4 Models)"
        print("="*60)
        print(f"MERGE CONFIGURATION - {merge_type}")
        print("="*60)
        print(f"Model A:          {args.A}")
        print(f"Model B:          {args.B}")
        print(f"Model C:          {args.C}")
        print(f"Model D:          {args.D}")
        print(f"Formula:          merged = C + {args.beta} * (B - A) + {1-args.beta} * (D - C)")
        print(f"Beta:             {args.beta}")
        print(f"Output:           {args.output}")
        print("="*60)
    elif args.A and args.B and args.lora_path:
        # General LoRA merge mode
        merge_type = "General LoRA Merge"
        print("="*60)
        print(f"MERGE CONFIGURATION - {merge_type}")
        print("="*60)
        print(f"Model A:          {args.A}")
        print(f"Model B:          {args.B}")
        print(f"LoRA adapter:     {args.lora_path}")
        print(f"Formula:          merged = A + {args.beta} * (B - A) + {1-args.beta} * lora_adapter")
        print(f"Beta:             {args.beta}")
        print(f"Output:           {args.output}")
        print("="*60)
    else:
        # Full model merge mode
        merge_type = "Full Model Merge"
        print("="*60)
        print(f"MERGE CONFIGURATION - {merge_type}")
        print("="*60)
        print(f"Base model:       {args.base}")
        print(f"Instruct model:   {args.instruct}")
        print(f"Fine-tuned model: {args.fine_tuned}")
        print(f"Formula:          merged = instruct + {args.alpha} * (fine_tuned - base)")
        print(f"Alpha:            {args.alpha}")
        print(f"Output:           {args.output}")
        print("="*60)
    
    print(f"\nStarting {merge_type.lower()}...")
    
    # Perform merge based on type
    if args.lora_path and args.lora_base:
        # Simple LoRA merge
        merged_model, tokenizer = simple_lora_merge(args, device, dtype)
        merge_info = {
            "merge_type": "simple_lora",
            "base_model": args.lora_base,
            "lora_adapter": args.lora_path,
            "formula": "merged = base + lora_adapter"
        }
    elif args.lora_path:
        # Complex LoRA merge
        merged_model, tokenizer = merge_lora_models(args, device, dtype)
        merge_info = {
            "merge_type": "complex_lora",
            "base_model": args.base,
            "instruct_model": args.instruct,
            "lora_adapter": args.lora_path,
            "alpha": args.alpha,
            "formula": "merged = instruct + alpha * (lora_merged_base - base)"
        }
    elif args.A and args.B and args.C and args.D:
        # General merge
        merged_model, tokenizer = merge_general_models(args, device, dtype)
        merge_info = {
            "merge_type": "general",
            "model_A": args.A,
            "model_B": args.B,
            "model_C": args.C,
            "model_D": args.D,
            "beta": args.beta,
            "formula": "merged = C + beta * (B - A) + (1 - beta) * (D - C)"
        }
    else:
        # Full model merge
        merged_model, tokenizer = merge_full_models(args, device, dtype)
        merge_info = {
            "merge_type": "full_model",
            "base_model": args.base,
            "instruct_model": args.instruct,
            "fine_tuned_model": args.fine_tuned,
            "alpha": args.alpha,
            "formula": "merged = instruct + alpha * (fine_tuned - base)"
        }
    
    # Save merged model
    print(f"Saving merged model to {args.output}...")
    merged_model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    
    # Save merge metadata
    metadata = {
        "merge_info": {
            **merge_info,
            "dtype": str(dtype),
            "device": device,
            "output_path": args.output
        }
    }
    
    import json
    with open(f"{args.output}/merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✅ {merge_type.upper()} COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Output: {args.output}")
    print(f"🧮 Formula: {merge_info['formula']}")
    if args.lora_path and args.lora_base:
        print(f"🏷️  Tokenizer: From {args.lora_base}")
    elif args.A and args.B and args.C and args.D:
        print(f"🏷️  Tokenizer: From {args.C}")
    else:
        print(f"🏷️  Tokenizer: From {args.instruct}")
    print(f"💾 Precision: {dtype}")
    print(f"🔧 Merge Type: {merge_type}")
    print("="*60)

if __name__ == "__main__":
    main()
