# Analysis: Accuracy vs MHD Tradeoff

## The Problem

There is a **fundamental contradiction** in the current implementation:

### Current Behavior

1. **Training**: Uses `CrossEntropyLoss` which encourages **exact species-level matches**
   ```python
   criterion = nn.CrossEntropyLoss()  # Line 90 in train.py
   ```

2. **Prediction**: Uses `argmax` - always picks the highest probability class
   ```python
   preds = probs.argmax(dim=1)  # Line 180 in train.py
   ```

3. **Model Selection**: Saves best model based on **accuracy**, NOT MHD
   ```python
   # Save best by val accuracy (simple criterion)
   if val_acc > best_val:  # Line 239 in train.py
       best_val = val_acc
       save_checkpoint(model, best_ckpt_path)
   ```

4. **MHD**: Only computed for **logging**, not used in training or model selection

### The Contradiction

- **Accuracy metric**: Only counts exact matches. If model predicts "Sebastolobus" (genus) when true label is "Sebastolobus alascanus" (species), accuracy = 0.

- **MHD metric**: Measures taxonomic distance. Predicting "Sebastolobus" when true is "Sebastolobus alascanus" would have low MHD (distance = 2, since same genus).

- **Biological reality**: Sometimes predicting a higher taxonomic rank (genus, family) when uncertain is more scientifically sound than guessing a wrong species.

### Example Scenario

**True label**: "Sebastolobus alascanus" (species)

**Scenario A**: Model predicts "Sebastolobus" (genus)
- Accuracy: 0 ❌
- MHD: 2 ✅ (close taxonomically)
- Biologically reasonable: Yes ✅

**Scenario B**: Model predicts "Apostichopus leukothele" (different species)
- Accuracy: 0 ❌  
- MHD: 8+ ❌ (far taxonomically)
- Biologically reasonable: No ❌

Both scenarios have accuracy = 0, but Scenario A is much better for MHD and biological interpretation!

## Current Code Limitations

1. ❌ **No hierarchical loss**: Training doesn't consider taxonomic relationships
2. ❌ **No uncertainty handling**: Model always picks argmax, even when uncertain
3. ❌ **Wrong optimization target**: Optimizes for accuracy, but competition evaluates on MHD
4. ❌ **No hierarchical prediction**: Can't predict higher taxonomic levels when uncertain

## Potential Solutions

### Solution 1: Hierarchical Loss Function

Replace `CrossEntropyLoss` with a hierarchical-aware loss that penalizes predictions based on taxonomic distance:

```python
def hierarchical_loss(logits, labels, taxonomy, alpha=0.5):
    """
    Combined loss: CrossEntropy + Hierarchical Distance
    alpha: weight for hierarchical component
    """
    ce_loss = F.cross_entropy(logits, labels)
    
    # Get predictions
    preds = logits.argmax(dim=1)
    
    # Compute hierarchical distances
    hd_loss = compute_hierarchical_distance_loss(preds, labels, taxonomy)
    
    return (1 - alpha) * ce_loss + alpha * hd_loss
```

### Solution 2: Model Selection by MHD

Change checkpoint saving to use MHD instead of accuracy:

```python
# Save best by MHD (lower is better)
if mhd is not None and mhd < best_mhd:
    best_mhd = mhd
    save_checkpoint(model, best_ckpt_path)
```

### Solution 3: Hierarchical Prediction Strategy

Instead of always using argmax, use a hierarchical prediction strategy:

```python
def hierarchical_predict(probs, taxonomy, confidence_threshold=0.7):
    """
    If max probability < threshold, predict at higher taxonomic level
    """
    max_prob = probs.max()
    if max_prob < confidence_threshold:
        # Find best prediction at genus/family level
        return predict_at_higher_level(probs, taxonomy)
    else:
        return probs.argmax(dim=1)
```

### Solution 4: Multi-Level Output

Modify model to output predictions at multiple taxonomic levels simultaneously, then select based on confidence.

## Recommendation

**Immediate fix**: Change model selection to use MHD instead of accuracy (Solution 2).

**Better fix**: Implement hierarchical loss function (Solution 1) to train the model to be aware of taxonomic relationships.

**Best fix**: Combine Solutions 1, 2, and 3 for a complete hierarchical-aware system.

