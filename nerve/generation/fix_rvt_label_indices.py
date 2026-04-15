#!/usr/bin/env python3
"""
Fix RVT label indices issue where:
1. Labels are not sorted by timestamp
2. objframe_idx_2_label_idx doesn't correctly map timestamps to label ranges

This script fixes the labels.npz file to ensure:
- Labels are sorted by timestamp
- objframe_idx_2_label_idx correctly maps each unique timestamp to its first label index
- Multiple labels at the same timestamp are contiguous
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def check_sequence(seq_path: Path) -> dict:
    """Check a sequence for label ordering issues."""
    labels_dir = seq_path / 'labels_v2'
    
    status = {
        'path': seq_path,
        'valid': True,
        'issues': [],
        'labels_unsorted': False,
        'multi_ts_frames': 0,
    }
    
    if not labels_dir.is_dir():
        status['valid'] = False
        status['issues'].append('Missing labels_v2 directory')
        return status
    
    labels_npz = labels_dir / 'labels.npz'
    if not labels_npz.is_file():
        status['valid'] = False
        status['issues'].append('Missing labels.npz')
        return status
    
    try:
        label_data = np.load(str(labels_npz))
        objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
        labels = label_data['labels']
        label_timestamps = np.load(str(labels_dir / 'timestamps_us.npy'))
    except Exception as e:
        status['valid'] = False
        status['issues'].append(f'Error loading data: {e}')
        return status
    
    if len(labels) == 0:
        status['issues'].append('Empty labels')
        return status
    
    # Check if labels are sorted by timestamp
    label_ts = labels['t']
    is_sorted = np.all(label_ts[:-1] <= label_ts[1:])
    if not is_sorted:
        status['labels_unsorted'] = True
        status['issues'].append('Labels not sorted by timestamp')
    
    # Check if any frames have labels with multiple timestamps
    for i in range(len(objframe_idx_2_label_idx)):
        is_last = (i == len(objframe_idx_2_label_idx) - 1)
        from_idx = objframe_idx_2_label_idx[i]
        to_idx = len(labels) if is_last else objframe_idx_2_label_idx[i + 1]
        
        if from_idx >= len(labels):
            continue
            
        frame_labels = labels[from_idx:to_idx]
        if len(frame_labels) > 0:
            unique_t = np.unique(frame_labels['t'])
            if len(unique_t) > 1:
                status['multi_ts_frames'] += 1
    
    if status['multi_ts_frames'] > 0:
        status['issues'].append(f'{status["multi_ts_frames"]} frames have labels with multiple timestamps')
    
    status['valid'] = len(status['issues']) == 0
    return status


def fix_sequence(seq_path: Path, verbose: bool = False) -> bool:
    """Fix label ordering and indexing for a sequence."""
    labels_dir = seq_path / 'labels_v2'
    
    try:
        label_data = np.load(str(labels_dir / 'labels.npz'))
        old_objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
        labels = label_data['labels']
        label_timestamps = np.load(str(labels_dir / 'timestamps_us.npy'))
    except Exception as e:
        print(f'  Error loading {seq_path.name}: {e}')
        return False
    
    if len(labels) == 0:
        if verbose:
            print(f'  {seq_path.name}: No labels to fix')
        return True
    
    # Step 1: Sort labels by timestamp
    sorted_indices = np.argsort(labels['t'])
    sorted_labels = labels[sorted_indices]
    
    # Step 2: Build new objframe_idx_2_label_idx
    # This maps each unique timestamp (from label_timestamps) to the first label index
    
    # Create a mapping from timestamp to list of label indices
    ts_to_label_indices = defaultdict(list)
    for idx, lbl in enumerate(sorted_labels):
        ts_to_label_indices[lbl['t']].append(idx)
    
    # Build the new mapping
    new_objframe_idx_2_label_idx = []
    for ts in label_timestamps:
        if ts in ts_to_label_indices:
            # Point to the first label with this timestamp
            new_objframe_idx_2_label_idx.append(min(ts_to_label_indices[ts]))
        else:
            # This shouldn't happen, but handle gracefully
            if verbose:
                print(f'  Warning: timestamp {ts} not found in labels for {seq_path.name}')
            # Use the previous index or 0
            prev_idx = new_objframe_idx_2_label_idx[-1] if new_objframe_idx_2_label_idx else 0
            new_objframe_idx_2_label_idx.append(prev_idx)
    
    new_objframe_idx_2_label_idx = np.array(new_objframe_idx_2_label_idx, dtype=np.int64)
    
    # Verify the fix
    issues_after = 0
    for i in range(len(new_objframe_idx_2_label_idx)):
        is_last = (i == len(new_objframe_idx_2_label_idx) - 1)
        from_idx = new_objframe_idx_2_label_idx[i]
        to_idx = len(sorted_labels) if is_last else new_objframe_idx_2_label_idx[i + 1]
        
        if from_idx >= len(sorted_labels):
            continue
            
        frame_labels = sorted_labels[from_idx:to_idx]
        if len(frame_labels) > 0:
            unique_t = np.unique(frame_labels['t'])
            if len(unique_t) > 1:
                issues_after += 1
    
    if issues_after > 0:
        print(f'  Warning: {seq_path.name} still has {issues_after} frames with multi-timestamp issues after fix')
        return False
    
    # Save the fixed data
    np.savez(
        str(labels_dir / 'labels.npz'),
        labels=sorted_labels,
        objframe_idx_2_label_idx=new_objframe_idx_2_label_idx
    )
    
    if verbose:
        print(f'  Fixed {seq_path.name}: sorted {len(sorted_labels)} labels, '
              f'rebuilt {len(new_objframe_idx_2_label_idx)} index mappings')
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix RVT label indices')
    parser.add_argument('dataset_root', type=str, 
                        help='Root directory of the RVT dataset')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to fix (train/val/test)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only check for issues, do not fix')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    split_path = dataset_root / args.split
    
    if not split_path.is_dir():
        print(f'Error: {split_path} is not a directory')
        return
    
    sequences = sorted([d for d in split_path.iterdir() 
                        if d.is_dir() and d.name.startswith('sequence_')])
    
    print(f'Checking {len(sequences)} sequences in {split_path}')
    print()
    
    # Check all sequences
    results = {
        'valid': [],
        'labels_unsorted': [],
        'multi_ts_frames': [],
        'other_issues': [],
    }
    
    for seq_path in tqdm(sequences, desc='Checking'):
        status = check_sequence(seq_path)
        
        if status['valid']:
            results['valid'].append(seq_path)
        elif status['labels_unsorted'] or status['multi_ts_frames'] > 0:
            if status['labels_unsorted']:
                results['labels_unsorted'].append(seq_path)
            if status['multi_ts_frames'] > 0:
                results['multi_ts_frames'].append(seq_path)
        else:
            results['other_issues'].append((seq_path, status['issues']))
    
    # Summary
    print()
    print('=' * 60)
    print('SUMMARY'.center(60))
    print('=' * 60)
    print(f'Valid sequences:              {len(results["valid"])}')
    print(f'Labels unsorted:              {len(results["labels_unsorted"])}')
    print(f'Multi-timestamp frames:       {len(results["multi_ts_frames"])}')
    print(f'Other issues:                 {len(results["other_issues"])}')
    print()
    
    # Sequences that need fixing
    needs_fix = set(results['labels_unsorted']) | set(results['multi_ts_frames'])
    
    if args.dry_run:
        print('DRY RUN - no changes made')
        if needs_fix:
            print(f'\nSequences that would be fixed ({len(needs_fix)}):')
            for seq in sorted(needs_fix, key=lambda x: x.name):
                print(f'  - {seq.name}')
    else:
        if needs_fix:
            print(f'Fixing {len(needs_fix)} sequences...')
            fixed = 0
            failed = 0
            for seq_path in tqdm(sorted(needs_fix, key=lambda x: x.name), desc='Fixing'):
                if fix_sequence(seq_path, verbose=args.verbose):
                    fixed += 1
                else:
                    failed += 1
            
            print(f'\nFixed: {fixed}')
            print(f'Failed: {failed}')
            
            # Verify
            print('\nVerifying fixes...')
            all_fixed = True
            for seq_path in needs_fix:
                status = check_sequence(seq_path)
                if not status['valid']:
                    print(f'  {seq_path.name}: Still has issues: {status["issues"]}')
                    all_fixed = False
            
            if all_fixed:
                print('All sequences verified successfully!')
        else:
            print('No sequences need fixing.')
    
    if results['other_issues']:
        print('\nSequences with other issues:')
        for seq_path, issues in results['other_issues']:
            print(f'  {seq_path.name}: {issues}')


if __name__ == '__main__':
    main()







