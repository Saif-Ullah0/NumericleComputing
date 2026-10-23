import struct
import numpy as np

def ieee16_analysis(val):
    """
    Analyze a float in IEEE-754 half-precision (16-bit) format.
    Extracts sign, exponent, mantissa; handles subnormals;
    computes chopping vs rounding errors.
    """
    sign = 0 if val >= 0 else 1
    x = abs(val)
    
    if x == 0:
        return {'sign': sign, 'exponent': 0, 'mantissa': 0, 
                'value': 0.0, 'type': 'zero'}
    
    # Find exponent
    import math
    e = math.floor(math.log2(x))
    exponent = e + 15
    
    # Normalize
    frac = x / (2**e) - 1.0
    
    # Mantissa bits
    mantissa_float = frac * 1024
    
    # Rounding and chopping
    mant_round = int(round(mantissa_float))
    mant_chop = int(mantissa_float)
    
    # Handle overflow
    if mant_round >= 1024:
        mant_round = 0
        exponent += 1
    if exponent >= 31:
        return {'sign': sign, 'exponent': 31, 'mantissa': 0,
                'value': float('inf') if sign == 0 else float('-inf'),
                'type': 'infinity'}
    
    # Check subnormal
    if exponent <= 0:
        exponent = 0
        mant_round = int(round(x / (2**-14) * 1024))
        mant_chop = int(x / (2**-14) * 1024)
        if mant_round >= 1024:
            mant_round = 0
            exponent = 1
        if mant_chop >= 1024:
            mant_chop = 0
            exponent = 1
    
    # Reconstruct values
    if exponent == 0:
        val_round = (-1)**sign * (mant_round / 1024) * 2**-14
        val_chop = (-1)**sign * (mant_chop / 1024) * 2**-14
    else:
        val_round = (-1)**sign * (1 + mant_round/1024) * 2**(exponent-15)
        val_chop = (-1)**sign * (1 + mant_chop/1024) * 2**(exponent-15)
    
    return {
        'sign': sign,
        'exponent': exponent,
        'mantissa_round': mant_round,
        'mantissa_chop': mant_chop,
        'value_round': val_round,
        'value_chop': val_chop,
        'type': 'normalized' if exponent > 0 else 'subnormal'
    }

# Test with given values
x = -0.00008544921875
result = ieee16_analysis(x)
print("=== x = -0.00008544921875 ===")
print(f"Sign: {result['sign']}")
print(f"Exponent: {result['exponent']} -> {result['exponent']:05b}")
print(f"Mantissa (round): {result['mantissa_round']:010b}")
print(f"Mantissa (chop):  {result['mantissa_chop']:010b}")
print(f"Type: {result['type']}")

y = 12.3876953125
result_y = ieee16_analysis(y)
print("\n=== y = 12.3876953125 ===")
print(f"Sign: {result_y['sign']}")
print(f"Exponent: {result_y['exponent']} -> {result_y['exponent']:05b}")
print(f"Mantissa (round): {result_y['mantissa_round']:010b}")
print(f"Mantissa (chop):  {result_y['mantissa_chop']:010b}")
print(f"Value (round): {result_y['value_round']}")
print(f"Value (chop):  {result_y['value_chop']}")

# Errors
true = y
approx_round = result_y['value_round']
approx_chop = result_y['value_chop']

abs_round = abs(true - approx_round)
abs_chop = abs(true - approx_chop)
rel_round = abs_round / abs(true)
rel_chop = abs_chop / abs(true)

print(f"\n=== Errors for y ===")
print(f"Rounding: abs={abs_round:.10f}, rel={rel_round:.10e}, %={rel_round*100:.6f}%")
print(f"Chopping: abs={abs_chop:.10f}, rel={rel_chop:.10e}, %={rel_chop*100:.6f}%")