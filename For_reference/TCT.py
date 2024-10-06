import numpy as np

def calculate_tct(mean_A, mean_B, std_A, std_B):
    """
    Calculate Tissue Contrast T-score (TCT) between two tissues.
    
    Parameters:
    mean_A (float): Mean intensity of tissue A
    mean_B (float): Mean intensity of tissue B
    std_A (float): Standard deviation of intensity of tissue A
    std_B (float): Standard deviation of intensity of tissue B
    
    Returns:
    float: TCT score
    """
    tct = abs(mean_A - mean_B) / np.sqrt(std_A**2 + std_B**2)
    return tct

# Example values for tissue A and B
mean_A = 150.0  # mean intensity of tissue A
mean_B = 120.0  # mean intensity of tissue B
std_A = 10.0    # standard deviation of tissue A
std_B = 12.0    # standard deviation of tissue B

# Calculate TCT
tct_score = calculate_tct(mean_A, mean_B, std_A, std_B)

print(f"Tissue Contrast T-score (TCT): {tct_score:.4f}")
