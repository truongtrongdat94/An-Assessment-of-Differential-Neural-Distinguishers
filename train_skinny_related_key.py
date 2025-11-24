"""
Training script for SKINNY-64/64 Related-Key Differential Neural Distinguisher

This script trains a neural distinguisher for related-key differential attacks
on SKINNY-64/64 cipher using both plaintext difference (ΔP) and key difference (ΔK).

Usage:
    python train_skinny_related_key.py
"""

from train_nets import train_distinguisher_related_key
from cipher.skinny import Skinny

# =============================================
# CONFIGURATION
# =============================================

# Cipher configuration
N_ROUNDS = 7  # Number of rounds for SKINNY-64/64

# Differential characteristics
DIFF_P = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]  # ΔP (plaintext difference)
DIFF_K = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # ΔK (key difference)

# Training configuration
N_TRAIN_SAMPLES = 10**7    # 10 million training samples
N_VAL_SAMPLES = 10**6      # 1 million validation samples
N_EPOCHS = 80              # Number of training epochs

# Network hyperparameters (từ repo gốc cho SKINNY)
DEPTH = 10                 # ResNet depth
N_NEURONS = 64             # Neurons in dense layers
KERNEL_SIZE = 3            # Convolution kernel size
N_FILTERS = 32             # Number of convolutional filters
REG_PARAM = 10**-5         # L2 regularization parameter
LR_HIGH = 0.002            # High learning rate
LR_LOW = 0.0001            # Low learning rate

# Data preprocessing
CALC_BACK = 2              # Variant 2: revert MixColumns, ShiftRows, Constants, SBox(8-15)

# =============================================
# MAIN TRAINING
# =============================================

if __name__ == "__main__":
    print("="*80)
    print("SKINNY-64/64 RELATED-KEY DIFFERENTIAL NEURAL DISTINGUISHER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Cipher: SKINNY-64/64 with {N_ROUNDS} rounds")
    print(f"  ΔP (plaintext difference): {DIFF_P}")
    print(f"  ΔK (key difference):       {DIFF_K}")
    print(f"  Training samples:  {N_TRAIN_SAMPLES:,}")
    print(f"  Validation samples: {N_VAL_SAMPLES:,}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  calc_back variant: {CALC_BACK}")
    print()
    
    # Initialize cipher
    skinny = Skinny(n_rounds=N_ROUNDS)
    
    # Train the related-key distinguisher
    train_distinguisher_related_key(
        skinny, 
        diff_p=DIFF_P,
        diff_k=DIFF_K,
        n_train_samples=N_TRAIN_SAMPLES,
        n_val_samples=N_VAL_SAMPLES,
        n_epochs=N_EPOCHS,
        depth=DEPTH,
        n_neurons=N_NEURONS,
        kernel_size=KERNEL_SIZE,
        n_filters=N_FILTERS,
        reg_param=REG_PARAM,
        lr_high=LR_HIGH,
        lr_low=LR_LOW,
        calc_back=CALC_BACK
    )
