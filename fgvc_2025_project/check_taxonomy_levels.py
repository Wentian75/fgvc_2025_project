#!/usr/bin/env python3
"""
Script to analyze how many taxonomic hierarchy levels are provided in the dataset.
"""
import json
from pathlib import Path
import csv

def analyze_dataset():
    # Check annotations.csv
    csv_path = Path("data/train/annotations.csv")
    print("=" * 60)
    print("ANALYZING DATASET STRUCTURE")
    print("=" * 60)
    
    print("\n1. ANNOTATIONS.CSV STRUCTURE:")
    print("-" * 60)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        print(f"Columns in CSV: {reader.fieldnames}")
        print(f"\nSample rows:")
        for i, row in enumerate(reader):
            if i < 5:
                print(f"  Row {i+1}: path={row['path'][:50]}..., label={row['label']}")
            else:
                break
    
    # Check taxonomy cache
    tax_cache_path = Path("cache/taxonomy_reference.json")
    if tax_cache_path.exists():
        print("\n2. TAXONOMY RESOLUTION (from WoRMS API cache):")
        print("-" * 60)
        with open(tax_cache_path, 'r') as f:
            tax = json.load(f)
        
        all_levels = []
        for concept, path in tax.items():
            non_null_levels = len([x for x in path if x is not None])
            all_levels.append(non_null_levels)
        
        print(f"Total concepts: {len(tax)}")
        print(f"Min levels resolved: {min(all_levels)}")
        print(f"Max levels resolved: {max(all_levels)}")
        print(f"Average levels resolved: {sum(all_levels)/len(all_levels):.2f}")
        print(f"\nBreakdown:")
        print(f"  Concepts with 7 levels (full): {sum(1 for x in all_levels if x == 7)} ({100*sum(1 for x in all_levels if x == 7)/len(all_levels):.1f}%)")
        print(f"  Concepts with 6 levels: {sum(1 for x in all_levels if x == 6)} ({100*sum(1 for x in all_levels if x == 6)/len(all_levels):.1f}%)")
        print(f"  Concepts with 5 levels: {sum(1 for x in all_levels if x == 5)} ({100*sum(1 for x in all_levels if x == 5)/len(all_levels):.1f}%)")
        print(f"  Concepts with 4 levels: {sum(1 for x in all_levels if x == 4)} ({100*sum(1 for x in all_levels if x == 4)/len(all_levels):.1f}%)")
        print(f"  Concepts with <4 levels: {sum(1 for x in all_levels if x < 4)} ({100*sum(1 for x in all_levels if x < 4)/len(all_levels):.1f}%)")
        
        print(f"\nSample entries:")
        for i, (concept, path) in enumerate(list(tax.items())[:5]):
            non_null = [x for x in path if x is not None]
            print(f"  {concept}: {len(non_null)} levels - {non_null}")
    
    print("\n3. SUMMARY:")
    print("-" * 60)
    print("❌ The dataset CSV provides ONLY 1 level: the concept/species name")
    print("✅ The code resolves 7-level taxonomy via WoRMS API at runtime")
    print("✅ The resolved taxonomy is cached in cache/taxonomy_reference.json")
    print("⚠️  Not all concepts have full 7 levels (some are higher taxonomic ranks)")
    print("\nExpected 7 levels: Kingdom → Phylum → Class → Order → Family → Genus → Species")
    print("=" * 60)

if __name__ == "__main__":
    analyze_dataset()

