#!/usr/bin/env python3
"""
Fix corrupted RVT sequence data.

This script repairs objframe_idx_2_repr_idx.npy files that have non-monotonic
indices due to incorrect chunk merging during dataset generation.

The fix works by:
1. Loading the timestamps from both event representations and labels
2. Building a proper first-occurrence mapping
3. Regenerating objframe_idx_2_repr_idx with monotonically increasing indices

Usage:
    python fix_rvt_sequences.py /path/to/dataset [--split train] [--ev-repr-name stacked_histogram_dt=16_nbins=10]
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def check_sequence(seq_path: Path, ev_repr_name: str) -> dict:
    """
    Check a sequence for issues.
    
    Returns dict with:
        - valid: bool
        - issue: str or None
        - details: dict with diagnostic info
    """
    result = {
        'valid': True,
        'issue': None,
        'details': {}
    }
    
    ev_repr_dir = seq_path / 'event_representations_v2' / ev_repr_name
    labels_dir = seq_path / 'labels_v2'
    
    # Check required files exist
    objframe_file = ev_repr_dir / 'objframe_idx_2_repr_idx.npy'
    timestamps_file = ev_repr_dir / 'timestamps_us.npy'
    labels_file = labels_dir / 'labels.npz'
    
    if not objframe_file.exists():
        result['valid'] = False
        result['issue'] = 'missing_objframe_file'
        return result
    
    if not timestamps_file.exists():
        result['valid'] = False
        result['issue'] = 'missing_timestamps_file'
        return result
    
    # Load data
    objframe_idx = np.load(str(objframe_file))
    result['details']['objframe_len'] = len(objframe_idx)
    
    # Check for empty
    if len(objframe_idx) == 0:
        result['valid'] = False
        result['issue'] = 'empty_labels'
        return result
    
    # Check for non-monotonic
    if len(objframe_idx) > 1:
        diffs = np.diff(objframe_idx)
        if np.any(diffs < 0):
            result['valid'] = False
            result['issue'] = 'non_monotonic'
            result['details']['min_diff'] = int(diffs.min())
            result['details']['num_decreasing'] = int(np.sum(diffs < 0))
            return result
    
    return result


def fix_sequence(seq_path: Path, ev_repr_name: str, verbose: bool = False) -> bool:
    """
    Fix a corrupted sequence.
    
    Returns True if fixed successfully, False otherwise.
    """
    ev_repr_dir = seq_path / 'event_representations_v2' / ev_repr_name
    labels_dir = seq_path / 'labels_v2'
    
    timestamps_file = ev_repr_dir / 'timestamps_us.npy'
    objframe_file = ev_repr_dir / 'objframe_idx_2_repr_idx.npy'
    labels_ts_file = labels_dir / 'timestamps_us.npy'
    
    try:
        # Load event representation timestamps
        repr_timestamps = np.load(str(timestamps_file))
        
        # Load label timestamps
        label_timestamps = np.load(str(labels_ts_file))
        
        if len(label_timestamps) == 0:
            if verbose:
                print(f"  {seq_path.name}: No labels, skipping")
            return False
        
        # Build FIRST occurrence mapping (key fix)
        ts_to_first_idx = {}
        for idx, ts in enumerate(repr_timestamps):
            if ts not in ts_to_first_idx:
                ts_to_first_idx[ts] = idx
        
        # Map label timestamps to repr indices using first occurrence
        new_objframe_idx = []
        for ts in label_timestamps:
            if ts in ts_to_first_idx:
                new_objframe_idx.append(ts_to_first_idx[ts])
            else:
                # Label timestamp not found in repr timestamps - find nearest
                idx = np.searchsorted(np.sort(list(ts_to_first_idx.keys())), ts)
                sorted_keys = sorted(ts_to_first_idx.keys())
                if idx >= len(sorted_keys):
                    idx = len(sorted_keys) - 1
                nearest_ts = sorted_keys[idx]
                new_objframe_idx.append(ts_to_first_idx[nearest_ts])
        
        new_objframe_idx = np.array(new_objframe_idx, dtype=np.int64)
        
        # Ensure monotonicity by sorting
        if len(new_objframe_idx) > 1:
            is_monotonic = np.all(np.diff(new_objframe_idx) >= 0)
            if not is_monotonic:
                # Sort by repr index
                sorted_pairs = sorted(zip(label_timestamps, new_objframe_idx), key=lambda x: x[1])
                sorted_label_ts = np.array([p[0] for p in sorted_pairs], dtype=np.int64)
                new_objframe_idx = np.array([p[1] for p in sorted_pairs], dtype=np.int64)
                
                # Also update label_timestamps file
                np.save(str(labels_ts_file), sorted_label_ts)
                
                if verbose:
                    print(f"  {seq_path.name}: Reordered labels by repr index")
        
        # Save fixed objframe_idx
        np.save(str(objframe_file), new_objframe_idx)
        
        # Verify fix
        final_check = np.all(np.diff(new_objframe_idx) >= 0) if len(new_objframe_idx) > 1 else True
        if not final_check:
            if verbose:
                print(f"  {seq_path.name}: FIX FAILED - still non-monotonic")
            return False
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"  {seq_path.name}: Error - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix corrupted RVT sequence data')
    parser.add_argument('dataset_path', help='Path to dataset root')
    parser.add_argument('--split', default='train', help='Split to fix (train/val/test)')
    parser.add_argument('--ev-repr-name', default='stacked_histogram_dt=16_nbins=10',
                       help='Event representation name')
    parser.add_argument('--dry-run', action='store_true', help='Check only, don\'t fix')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    split_path = dataset_path / args.split
    
    if not split_path.exists():
        print(f"Error: Split path not found: {split_path}")
        return 1
    
    # Find all sequences
    sequences = sorted([d for d in split_path.iterdir() 
                       if d.is_dir() and d.name.startswith('sequence_')])
    
    print(f"Checking {len(sequences)} sequences in {split_path}")
    print(f"Event representation: {args.ev_repr_name}")
    print()
    
    # Check all sequences
    issues = {
        'valid': [],
        'empty_labels': [],
        'non_monotonic': [],
        'missing_files': [],
        'other': []
    }
    
    for seq_path in tqdm(sequences, desc='Checking'):
        result = check_sequence(seq_path, args.ev_repr_name)
        
        if result['valid']:
            issues['valid'].append(seq_path)
        elif result['issue'] == 'empty_labels':
            issues['empty_labels'].append(seq_path)
        elif result['issue'] == 'non_monotonic':
            issues['non_monotonic'].append(seq_path)
        elif result['issue'] in ('missing_objframe_file', 'missing_timestamps_file'):
            issues['missing_files'].append(seq_path)
        else:
            issues['other'].append(seq_path)
    
    # Report
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Valid sequences:     {len(issues['valid'])}")
    print(f"Empty (no labels):   {len(issues['empty_labels'])}")
    print(f"Non-monotonic:       {len(issues['non_monotonic'])}")
    print(f"Missing files:       {len(issues['missing_files'])}")
    print(f"Other issues:        {len(issues['other'])}")
    print()
    
    if args.dry_run:
        print("DRY RUN - no changes made")
        if issues['non_monotonic']:
            print("\nSequences that would be fixed:")
            for seq in issues['non_monotonic'][:10]:
                print(f"  - {seq.name}")
            if len(issues['non_monotonic']) > 10:
                print(f"  ... and {len(issues['non_monotonic']) - 10} more")
        return 0
    
    # Fix non-monotonic sequences
    if issues['non_monotonic']:
        print(f"\nFixing {len(issues['non_monotonic'])} non-monotonic sequences...")
        fixed = 0
        failed = 0
        
        for seq_path in tqdm(issues['non_monotonic'], desc='Fixing'):
            if fix_sequence(seq_path, args.ev_repr_name, verbose=args.verbose):
                fixed += 1
            else:
                failed += 1
        
        print()
        print(f"Fixed: {fixed}")
        print(f"Failed: {failed}")
        
        # Verify fixes
        print("\nVerifying fixes...")
        still_broken = 0
        for seq_path in issues['non_monotonic']:
            result = check_sequence(seq_path, args.ev_repr_name)
            if not result['valid']:
                still_broken += 1
                if args.verbose:
                    print(f"  {seq_path.name}: Still broken - {result['issue']}")
        
        if still_broken == 0:
            print("All sequences verified successfully!")
        else:
            print(f"WARNING: {still_broken} sequences still have issues")
    else:
        print("No sequences need fixing!")
    
    return 0


if __name__ == '__main__':
    exit(main())







