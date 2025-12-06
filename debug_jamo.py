"""Debug script to understand Jamo decomposition issues."""

import numpy as np
from jamo import h2j, j2hcj
from dataloader import load_hog_features

# Load class names
_, _, class_names = load_hog_features(
    feature_dir="features/hog-extended", shuffle=False, random_state=42
)

print("Class names and their Jamo decomposition:")
print("=" * 80)

for i, char in enumerate(class_names):
    try:
        jamo = j2hcj(h2j(char))
        cho = jamo[0] if len(jamo) > 0 else ""
        jung = jamo[1] if len(jamo) > 1 else ""
        jong = jamo[2] if len(jamo) > 2 else ""
        print(f"{i:2d}. {char:5s} -> [{cho}, {jung}, {jong}]")
    except Exception as e:
        print(f"{i:2d}. {char:5s} -> ERROR: {e}")

print("\n" + "=" * 80)
print("Checking specific characters:")
print("=" * 80)

test_chars = ["yeo", "i", "ji", "deul", "jeong"]
for char in test_chars:
    if char in class_names:
        idx = class_names.index(char)
        jamo = j2hcj(h2j(char))
        cho = jamo[0] if len(jamo) > 0 else ""
        jung = jamo[1] if len(jamo) > 1 else ""
        jong = jamo[2] if len(jamo) > 2 else ""
        print(f"{char:10s} (idx={idx:2d}) -> [{cho}, {jung}, {jong}]")
    else:
        print(f"{char:10s} NOT in class_names!")
