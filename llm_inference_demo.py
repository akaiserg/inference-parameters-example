"""
LLM Inference Demo: From Logits to Next Token (COMPLETE VERSION)

This module demonstrates step-by-step how a language model generates 
the next token from logits using temperature, softmax, BOTH top-k AND top-p 
filtering (like real LLM APIs), and sampling.

Based on the example from info.md: "The cat is" → predicting next token
"""

from typing import Dict, List, Tuple
import math
import argparse
import sys


# ============================================================================
# INITIAL DATA - The starting point after the model produces logits
# ============================================================================

# Context: "The cat is"
# The model has processed these tokens and produced raw logits for candidates

INITIAL_LOGITS = {
    'sleeping': 2.8,
    'eating': 2.3,
    'playing': 1.9,
    'sitting': 1.5,
    'jumping': 1.2
}


# ============================================================================
# STEP 1: APPLY TEMPERATURE
# ============================================================================

def apply_temperature(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """
    Apply temperature scaling to logits.
    
    Temperature controls the "randomness" of predictions by reshaping the
    distribution BEFORE converting to probabilities.
    
    Formula:
        adjusted_logit_i = logit_i / temperature
    
    Effect:
        - T < 1.0: Makes distribution SHARPER (more confident, peaked)
                   Example: T=0.5 makes strong preferences stronger
        
        - T = 1.0: NO CHANGE (neutral, original distribution)
        
        - T > 1.0: Makes distribution FLATTER (less confident, more uniform)
                   Example: T=2.0 makes all options more equal
    
    Args:
        logits: Dictionary mapping token names to their logit values
        temperature: Scaling factor (must be > 0)
                    Default is 1.0 (no change)
    
    Returns:
        Dictionary with temperature-adjusted logits
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    
    adjusted_logits = {}
    for token, logit in logits.items():
        adjusted_logit = logit / temperature
        adjusted_logits[token] = adjusted_logit
    
    return adjusted_logits


# ============================================================================
# STEP 2: COMPUTE SOFTMAX (Exponentiation + Normalization)
# ============================================================================

def compute_softmax(logits: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Convert logits to probabilities using the softmax function.
    
    This is a TWO-STEP process:
    
    Step 2a - Exponentiation:
        Convert all logits to positive weights using e^x
        Formula: weight_i = e^(logit_i)
    
    Step 2b - Normalization:
        Divide each weight by the sum of all weights
        Formula: probability_i = weight_i / Σ(weight_j)
    
    Args:
        logits: Dictionary mapping token names to their logit values
    
    Returns:
        Tuple of (weights_dict, probabilities_dict)
    """
    # Step 2a: Exponentiation - convert to positive weights
    weights = {}
    for token, logit in logits.items():
        weight = math.exp(logit)
        weights[token] = weight
    
    # Calculate the normalization constant (sum of all weights)
    Z = sum(weights.values())
    
    # Step 2b: Normalization - convert to probabilities
    probabilities = {}
    for token, weight in weights.items():
        probability = weight / Z
        probabilities[token] = probability
    
    return weights, probabilities


# ============================================================================
# STEP 3A: APPLY TOP-K FILTERING
# ============================================================================

def apply_top_k(probabilities: Dict[str, float], k: int) -> Dict[str, float]:
    """
    Keep only the top K most probable tokens (first filter).
    
    This filtering technique:
    1. Sorts tokens by probability (highest first)
    2. Keeps only the top K tokens
    3. Discards all others
    
    Note: Probabilities are NOT renormalized yet (happens after top-p).
    
    Args:
        probabilities: Dictionary mapping token names to their probabilities
        k: Number of top tokens to keep (must be > 0)
    
    Returns:
        Dictionary with only top-k tokens (not yet renormalized)
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    
    # Sort tokens by probability (highest first)
    token_prob_list = []
    for token, prob in probabilities.items():
        token_prob_list.append((token, prob))
    
    # Bubble sort (highest first)
    sorted_tokens = token_prob_list.copy()
    n = len(sorted_tokens)
    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_tokens[j][1] < sorted_tokens[j + 1][1]:
                sorted_tokens[j], sorted_tokens[j + 1] = sorted_tokens[j + 1], sorted_tokens[j]
    
    # Keep only top K tokens
    if k > len(sorted_tokens):
        k = len(sorted_tokens)
    
    top_k_tokens = sorted_tokens[:k]
    
    # Return as dictionary (not renormalized yet)
    filtered_probabilities = {}
    for token, prob in top_k_tokens:
        filtered_probabilities[token] = prob
    
    return filtered_probabilities


# ============================================================================
# STEP 3B: APPLY TOP-P FILTERING (NUCLEUS SAMPLING)
# ============================================================================

def apply_top_p(probabilities: Dict[str, float], p: float) -> Dict[str, float]:
    """
    Keep tokens until cumulative probability reaches p (second filter).
    
    This filtering technique:
    1. Sorts tokens by probability (highest first)
    2. Keeps adding tokens until cumulative probability >= p
    3. Discards remaining tokens
    
    Note: Probabilities are NOT renormalized yet (happens after both filters).
    
    Args:
        probabilities: Dictionary mapping token names to their probabilities
        p: Cumulative probability threshold (0 < p <= 1.0)
    
    Returns:
        Dictionary with only nucleus tokens (not yet renormalized)
    """
    if p <= 0 or p > 1.0:
        raise ValueError(f"p must be in range (0, 1.0], got {p}")
    
    # Sort tokens by probability (highest first)
    token_prob_list = []
    for token, prob in probabilities.items():
        token_prob_list.append((token, prob))
    
    # Bubble sort (highest first)
    sorted_tokens = token_prob_list.copy()
    n = len(sorted_tokens)
    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_tokens[j][1] < sorted_tokens[j + 1][1]:
                sorted_tokens[j], sorted_tokens[j + 1] = sorted_tokens[j + 1], sorted_tokens[j]
    
    # Keep tokens until cumulative probability reaches p
    nucleus_tokens = []
    cumulative_prob = 0.0
    
    for token, prob in sorted_tokens:
        nucleus_tokens.append((token, prob))
        cumulative_prob = cumulative_prob + prob
        
        if cumulative_prob >= p:
            break
    
    # Return as dictionary (not renormalized yet)
    filtered_probabilities = {}
    for token, prob in nucleus_tokens:
        filtered_probabilities[token] = prob
    
    return filtered_probabilities


# ============================================================================
# STEP 3C: RENORMALIZE
# ============================================================================

def renormalize(probabilities: Dict[str, float]) -> Dict[str, float]:
    """
    Renormalize probabilities so they sum to 1.0.
    
    After applying top-k and/or top-p filters, the remaining probabilities
    no longer sum to 1.0. This function rescales them.
    
    Args:
        probabilities: Dictionary with filtered probabilities
    
    Returns:
        Dictionary with renormalized probabilities (sum = 1.0)
    """
    total = sum(probabilities.values())
    
    if total == 0:
        raise ValueError("Cannot renormalize: sum of probabilities is 0")
    
    renormalized = {}
    for token, prob in probabilities.items():
        renormalized[token] = prob / total
    
    return renormalized


# ============================================================================
# STEP 4: SAMPLE TOKEN
# ============================================================================

def sample_token(probabilities: Dict[str, float], seed: int = None) -> Tuple[str, float, Dict[str, Tuple[float, float]]]:
    """
    Sample a token based on the probability distribution.
    
    This implements probabilistic sampling:
    1. Build cumulative probability ranges for each token
    2. Generate a random number between 0 and 1
    3. Find which token's range contains the random number
    4. Return that token
    
    Args:
        probabilities: Dictionary mapping token names to their probabilities
        seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (selected_token, random_number, cumulative_ranges)
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    # Build cumulative probability ranges
    cumulative_ranges = {}
    cumulative_sum = 0.0
    
    for token, prob in probabilities.items():
        range_start = cumulative_sum
        range_end = cumulative_sum + prob
        cumulative_ranges[token] = (range_start, range_end)
        cumulative_sum = range_end
    
    # Generate random number
    random_number = random.random()
    
    # Find which token's range contains the random number
    selected_token = None
    for token, (range_start, range_end) in cumulative_ranges.items():
        if range_start <= random_number < range_end:
            selected_token = token
            break
    
    # Fallback
    if selected_token is None:
        selected_token = list(probabilities.keys())[-1]
    
    return selected_token, random_number, cumulative_ranges


# ============================================================================
# MAIN DEMO
# ============================================================================

def main(temperature: float = 1.0, top_k: int = None, top_p: float = None, seed: int = None):
    """
    Main demonstration function that will orchestrate all steps.
    
    Args:
        temperature: Temperature value for scaling logits (default: 1.0)
        top_k: Number of top tokens to keep (default: None = no top-k filtering)
        top_p: Cumulative probability threshold (default: None = no top-p filtering)
        seed: Random seed for sampling (default: None = random)
    """
    print("=" * 70)
    print("🧠 LLM INFERENCE DEMO: Complete Pipeline (TOP-K + TOP-P + TEMP)")
    print("=" * 70)
    print()
    
    # Show configuration
    print("⚙️  Configuration:")
    print(f"   • Temperature: {temperature}")
    print(f"   • Top-K: {top_k if top_k else 'disabled (no filtering)'}")
    print(f"   • Top-P: {top_p if top_p else 'disabled (no filtering)'}")
    print(f"   • Seed: {seed if seed is not None else 'random'}")
    print()
    
    # Show the context
    print("📝 Input Context:")
    print('   "The cat is"')
    print()
    print("🎯 Goal: Predict the next token")
    print()
    
    # Step 0: Show initial logits
    print("-" * 70)
    print("STEP 0: Model Output (Raw Logits)")
    print("-" * 70)
    print()
    
    print("Token          | Logit")
    print("---------------|-------")
    for token, logit in INITIAL_LOGITS.items():
        print(f"{token:14} | {logit:.1f}")
    print()
    
    # Step 1: Apply Temperature
    print("-" * 70)
    print("STEP 1: Apply Temperature")
    print("-" * 70)
    print()
    
    print(f"🌡️  Temperature: {temperature}")
    print()
    
    adjusted_logits = apply_temperature(INITIAL_LOGITS, temperature)
    
    print("Token          | Original | After T={:.1f}".format(temperature))
    print("---------------|----------|------------")
    for token in INITIAL_LOGITS.keys():
        original = INITIAL_LOGITS[token]
        adjusted = adjusted_logits[token]
        print(f"{token:14} | {original:8.1f} | {adjusted:10.2f}")
    print()
    
    # Step 2: Compute Softmax
    print("-" * 70)
    print("STEP 2: Compute Softmax (Exponentiation + Normalization)")
    print("-" * 70)
    print()
    
    weights, probabilities = compute_softmax(adjusted_logits)
    
    print("Token          | Probability | Percentage")
    print("---------------|-------------|------------")
    for token in probabilities.keys():
        prob = probabilities[token]
        percentage = prob * 100
        print(f"{token:14} | {prob:11.3f} | {percentage:6.1f}%")
    
    prob_sum = sum(probabilities.values())
    print("               |             |")
    print(f"               | {prob_sum:11.3f} | {prob_sum*100:6.1f}%")
    print()
    
    # Step 3: Apply Filters (Top-K and/or Top-P)
    print("-" * 70)
    print("STEP 3: Apply Filters (Top-K → Top-P → Renormalize)")
    print("-" * 70)
    print()
    
    filtered_probabilities = probabilities.copy()
    
    # Step 3a: Apply Top-K if specified
    if top_k is not None:
        print(f"✂️  Step 3a: Applying Top-K (k={top_k})")
        print()
        
        filtered_probabilities = apply_top_k(filtered_probabilities, top_k)
        
        print(f"Kept {len(filtered_probabilities)} tokens after top-k filtering:")
        for token in filtered_probabilities.keys():
            print(f"  ✅ {token}")
        print()
    else:
        print("⊘  Step 3a: Top-K disabled (keeping all tokens)")
        print()
    
    # Step 3b: Apply Top-P if specified
    if top_p is not None:
        print(f"🎯 Step 3b: Applying Top-P (p={top_p})")
        print()
        
        # Show cumulative probabilities
        sorted_probs = sorted(filtered_probabilities.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        print("Cumulative probability check:")
        for token, prob in sorted_probs:
            cumulative = cumulative + prob
            status = "✅" if cumulative <= top_p or len([t for t, p in sorted_probs if cumulative - p < top_p]) == 0 else "❌"
            print(f"  {token:12} {prob:.3f} → cumulative: {cumulative:.3f} {status}")
        print()
        
        filtered_probabilities = apply_top_p(filtered_probabilities, top_p)
        
        print(f"Kept {len(filtered_probabilities)} tokens after top-p filtering")
        print()
    else:
        print("⊘  Step 3b: Top-P disabled (keeping current tokens)")
        print()
    
    # Step 3c: Renormalize
    print("📊 Step 3c: Renormalization")
    print()
    
    print("Token          | After Filters | After Renorm | Change")
    print("---------------|---------------|--------------|--------")
    
    original_filtered = filtered_probabilities.copy()
    filtered_probabilities = renormalize(filtered_probabilities)
    
    for token in filtered_probabilities.keys():
        before = original_filtered[token]
        after = filtered_probabilities[token]
        change = ((after - before) / before) * 100 if before > 0 else 0
        change_str = f"+{change:5.1f}%" if change >= 0 else f"{change:6.1f}%"
        print(f"{token:14} | {before:13.3f} | {after:12.3f} | {change_str}")
    
    final_sum = sum(filtered_probabilities.values())
    print("               |               |              |")
    print(f"               | Sum =         | {final_sum:12.3f} |")
    print()
    print("✅ Final probabilities sum to 1.0")
    print()
    
    # Summary of filtering
    print("📋 Filtering Summary:")
    print(f"   • Started with: {len(probabilities)} tokens")
    print(f"   • After filters: {len(filtered_probabilities)} tokens")
    print(f"   • Eliminated: {len(probabilities) - len(filtered_probabilities)} tokens")
    print()
    
    # Step 4: Sample Token
    print("-" * 70)
    print("STEP 4: Sample Token (Final Selection)")
    print("-" * 70)
    print()
    
    selected_token, random_number, cumulative_ranges = sample_token(filtered_probabilities, seed)
    
    print("Cumulative probability ranges:")
    print()
    print("Token          | Probability | Range")
    print("---------------|-------------|------------------")
    
    for token in filtered_probabilities.keys():
        prob = filtered_probabilities[token]
        range_start, range_end = cumulative_ranges[token]
        range_str = f"[{range_start:.3f} – {range_end:.3f})"
        print(f"{token:14} | {prob:11.3f} | {range_str}")
    print()
    
    print(f"🎯 Random number: {random_number:.3f}")
    print()
    
    print("Finding the selected token:")
    for token in filtered_probabilities.keys():
        range_start, range_end = cumulative_ranges[token]
        if token == selected_token:
            print(f"  ✅ {token:12} [{range_start:.3f} – {range_end:.3f}) ← SELECTED!")
        else:
            print(f"  ❌ {token:12} [{range_start:.3f} – {range_end:.3f})")
    print()
    
    # Final Result
    print("=" * 70)
    print("🎉 FINAL RESULT")
    print("=" * 70)
    print()
    print(f"Selected token: **{selected_token.upper()}**")
    print()
    print("Complete sentence:")
    print(f'  "The cat is {selected_token}"')
    print()
    print("=" * 70)
    print("✅ ALL STEPS COMPLETE!")
    print("=" * 70)
    print()
    
    # Complete summary
    print("📊 Complete Pipeline Summary:")
    print(f"  • Original logit: {INITIAL_LOGITS[selected_token]:.1f}")
    print(f"  • After temperature: {adjusted_logits[selected_token]:.2f}")
    print(f"  • After softmax: {probabilities[selected_token]:.3f} ({probabilities[selected_token]*100:.1f}%)")
    if selected_token in filtered_probabilities:
        print(f"  • After filtering: {filtered_probabilities[selected_token]:.3f} ({filtered_probabilities[selected_token]*100:.1f}%)")
    print(f"  • Random number: {random_number:.3f}")
    print(f"  • ✅ Selected: {selected_token}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Inference Demo: Complete pipeline with temperature, top-k, and top-p",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 llm_inference_demo.py                           # Defaults (T=1.0, no filters)
  python3 llm_inference_demo.py -t 0.5                    # Temperature only
  python3 llm_inference_demo.py -k 3                      # Top-k only
  python3 llm_inference_demo.py -p 0.9                    # Top-p only
  python3 llm_inference_demo.py -k 4 -p 0.9               # Both filters
  python3 llm_inference_demo.py -t 0.7 -k 3 -p 0.9 -s 42  # Full config (like real LLMs!)
  python3 llm_inference_demo.py -t 2.0 -k 5 -p 1.0        # Maximum randomness
        """
    )
    
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=1.0,
        help='Temperature for scaling logits (must be > 0). Default: 1.0'
    )
    
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=None,
        dest='top_k',
        help='Number of top tokens to keep (optional). Default: None (disabled)'
    )
    
    parser.add_argument(
        '-p', '--top-p',
        type=float,
        default=None,
        dest='top_p',
        help='Cumulative probability threshold (0 < p <= 1.0, optional). Default: None (disabled)'
    )
    
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional). Default: None (random)'
    )
    
    args = parser.parse_args()
    
    # Validate temperature
    if args.temperature <= 0:
        print("❌ Error: Temperature must be greater than 0")
        print(f"   You provided: {args.temperature}")
        sys.exit(1)
    
    # Validate top-k
    if args.top_k is not None and args.top_k <= 0:
        print("❌ Error: top-k must be greater than 0")
        print(f"   You provided: {args.top_k}")
        sys.exit(1)
    
    if args.top_k is not None and args.top_k > 5:
        print("❌ Error: top-k cannot be greater than number of tokens (5)")
        print(f"   You provided: {args.top_k}")
        sys.exit(1)
    
    # Validate top-p
    if args.top_p is not None and (args.top_p <= 0 or args.top_p > 1.0):
        print("❌ Error: top-p must be in range (0, 1.0]")
        print(f"   You provided: {args.top_p}")
        sys.exit(1)
    
    # Run the demo
    main(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)

