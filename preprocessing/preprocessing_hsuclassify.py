import mne
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm
from mne_icalabel import label_components
from mne.preprocessing import ICA

base_path = r"C:\School\Master's\HMI lab\Project Hsu\Data3"
file_numbers = range(41, 390)
file_pattern = "Function-{:03d}.cnt"
output_dir = r'C:\mnt\user-data\outputs\cbramod_data_official'

os.makedirs(output_dir, exist_ok=True)

# KEEP_CHANNELS = [
#     # Front Left (AL)
#     'AF3', 'F7', 'F5', 'F3', 'FP1', 'FP2', 'FPZ',
#     # Front Center (AM)
#     'F1', 'FZ', 'F2',
#     # Front Right (AR)
#     'AF4', 'F4', 'F6', 'F8', 'T7','T8',
#     # Central Left (CL)
#     'FT7', 'FC5', 'FC3', 'C3', 'C5', 'TP7', 'CP5', 'CP3',
#     # Central Center (CM)
#     'FC1', 'C1', 'CP1', 'FCZ', 'CZ', 'CPZ', 'FC2', 'C2', 'CP2',
#     # Central Right (CR)
#     'FC4', 'FC6', 'FT8', 'C4', 'C6', 'CP6', 'CP4', 'TP8',
#     # Posterior Left (PL)
#     'P3', 'P5', 'P7', 'PO7', 'PO5', 'PO3', 'O1',
#     # Posterior Center (PM)
#     'P1', 'PZ', 'P2', 'OZ', 'POZ',
#     # Posterior Right (PR)
#     'P4', 'P6', 'P8', 'PO8', 'PO6', 'PO4', 'O2' 
# ]

KEEP_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',  
    'P7', 'P3', 'PZ', 'P4', 'P8',  
    'O1', 'O2'
]

TARGET_SFREQ = 200
BANDPASS_LOW = 0.1   
BANDPASS_HIGH = 75    
NOTCH_FREQ = 60       

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

all_segments = []
successfully_processed = 0
failed_files = []

print("=" * 60)
print("EEG DATA PREPROCESSING PIPELINE")
print("=" * 60)

for file_num in tqdm(file_numbers, desc="Processing files"):
    file_name = file_pattern.format(file_num)
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(file_path):
        continue
    
    try:
        try:
            raw = mne.io.read_raw_ant(file_path, preload=True, verbose=False)
        except:
            raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
        
        # Keep only required channels
        available_channels = [ch for ch in KEEP_CHANNELS if ch in raw.ch_names]
        if len(available_channels) == 0:
            failed_files.append((file_name, "No valid channels"))
            continue
        
        # Apply preprocessing
        raw.pick(available_channels)
        raw.reorder_channels(available_channels)
        raw.resample(TARGET_SFREQ, verbose=False)
        # raw.notch_filter(NOTCH_FREQ, verbose=False)
        # raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH)

        # ICA
        # ica = ICA(n_components=None, 
        #         method='picard', 
        #         fit_params=dict(ortho=False, extended=True), 
        #         random_state=42, 
        #         max_iter='auto')
        
        # ica.fit(raw, verbose=False)
        # ic_labels = label_components(raw, ica, method='iclabel')
        # labels = ic_labels['labels']
        # probs = ic_labels['y_pred_proba']
        
        # exclude_idx = []
        # target_artifacts = ['eye', 'muscle', 'heart']
        # threshold = 0.90
        # for i, (label, prob) in enumerate(zip(labels, probs)):
        #     if label in target_artifacts and prob >= threshold:
        #         exclude_idx.append(i)
        # if exclude_idx:
        #     ica.exclude = exclude_idx
        #     ica.apply(raw, verbose=False) 

        # Extract events
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        trigger_samples = events[:, 0]
        trigger_codes = events[:, 2]
        
        # Convert event codes to integers
        id_to_desc = {v: k for k, v in event_dict.items()}
        trigger_codes_int = []
        for event_id in trigger_codes:
            desc = id_to_desc[event_id]
            try:
                trigger_codes_int.append(int(desc))
            except ValueError:
                trigger_codes_int.append(-1)
        trigger_codes_int = np.array(trigger_codes_int)
        
        # Find segments: [60-120] → [200,201]
        i = 0
        while i < len(trigger_codes_int):
            if trigger_codes_int[i] in range(60, 121):
                start_sample = trigger_samples[i]
                start_trigger = trigger_codes_int[i]
                
                j = i + 1
                while j < len(trigger_codes_int):
                    if trigger_codes_int[j] in [200, 201]:
                        end_sample = trigger_samples[j]
                        end_trigger = trigger_codes_int[j]
                        
                        start_time = start_sample / raw.info['sfreq']
                        end_time = end_sample / raw.info['sfreq']
                        duration = end_time - start_time
                        
                        # Extract segment
                        segment_raw = raw.copy().crop(tmin=start_time, tmax=end_time)
                        segment_data = segment_raw.get_data(units='uV')
                        
                        # Store segment
                        all_segments.append({
                            'data': segment_data,
                            'file_num': file_num,
                            'start_trigger': start_trigger,
                            'end_trigger': end_trigger,
                            'duration': duration,
                            'n_channels': segment_data.shape[0]
                        })
                        
                        i = j
                        break
                    j += 1
            i += 1
        
        successfully_processed += 1
        
    except Exception as e:
        raise e
        failed_files.append((file_name, str(e)))

print(f"\n{'=' * 60}")
print(f"PROCESSING SUMMARY")
print(f"{'=' * 60}")
print(f"Processed:  {successfully_processed} files")
print(f"Segments:   {len(all_segments)}")
print(f"Failed:     {len(failed_files)} files")

if len(failed_files) > 0:
    print(f"\nFirst 5 failures:")
    for fname, error in failed_files[:5]:
        print(f"  {fname}: {error}")

# Separate by class
correct_segments = [s for s in all_segments if s['end_trigger'] == 201]
incorrect_segments = [s for s in all_segments if s['end_trigger'] == 200]

print(f"\n{'=' * 60}")
print(f"CLASS DISTRIBUTION")
print(f"{'=' * 60}")
print(f"Correct (201):   {len(correct_segments)}")
print(f"Incorrect (200): {len(incorrect_segments)}")

# Balance classes
min_samples = min(len(correct_segments), len(incorrect_segments))
if len(correct_segments) > min_samples:
    np.random.seed(42)
    correct_segments = list(np.random.choice(correct_segments, size=min_samples, replace=False))
if len(incorrect_segments) > min_samples:
    np.random.seed(42)
    incorrect_segments = list(np.random.choice(incorrect_segments, size=min_samples, replace=False))

print(f"\nAfter balancing:")
print(f"  Correct:   {len(correct_segments)}")
print(f"  Incorrect: {len(incorrect_segments)}")

def create_patches_official(data, patch_size=200):
    C, T = data.shape
    
    if T < patch_size:
        pad_width = patch_size - T
        data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        T = patch_size 
    
    n_patches = T // patch_size
    
    T_trimmed = n_patches * patch_size
    data_trimmed = data[:, :T_trimmed]
    
    # Reshape to (C, n, patch_size)
    patches = data_trimmed.reshape(C, n_patches, patch_size)
    
    return patches

# Analyze data statistics
all_lengths = [s['data'].shape[1] / TARGET_SFREQ for s in correct_segments + incorrect_segments]
all_channels = [s['n_channels'] for s in correct_segments + incorrect_segments]

print(f"\n{'=' * 60}")
print(f"DATA STATISTICS")
print(f"{'=' * 60}")
print(f"Durations: {np.min(all_lengths):.1f}s - {np.max(all_lengths):.1f}s")
print(f"Median:    {np.median(all_lengths):.1f}s")
print(f"Channels:  {np.unique(all_channels)}")

# Determine target configuration
fixed_channels = int(np.min(all_channels))
median_patches = round(np.median(all_lengths))
print(f"\nTarget configuration:")
print(f"  Channels: {fixed_channels}")
print(f"  Patches:  {median_patches} (median)")

def standardize_to_target(patches, target_channels, target_patches):
    """Standardize patches to target shape."""
    C, n, patch_size = patches.shape
    
    # Standardize channels
    if C > target_channels:
        patches = patches[:target_channels, :, :]
    elif C < target_channels:
        pad_ch = target_channels - C
        padding = np.zeros((pad_ch, n, patch_size), dtype=patches.dtype)
        patches = np.concatenate([patches, padding], axis=0)
    
    # Standardize patches
    if n > target_patches:
        # Crop from center
        start = (n - target_patches) // 2
        patches = patches[:, start:start+target_patches, :]
    elif n < target_patches:
        # Pad with zeros
        pad_n = target_patches - n
        padding = np.zeros((target_channels, pad_n, patch_size), dtype=patches.dtype)
        patches = np.concatenate([patches, padding], axis=1)
    
    return patches

# Process all segments
X_correct = []
X_incorrect = []

print(f"\n{'=' * 60}")
print(f"CREATING PATCHES")
print(f"{'=' * 60}")

print("Processing correct segments...")
for seg in tqdm(correct_segments):
    patches = create_patches_official(seg['data'])
    if patches is not None:
        standardized = standardize_to_target(patches, fixed_channels, median_patches)
        X_correct.append(standardized)

print("Processing incorrect segments...")
for seg in tqdm(incorrect_segments):
    patches = create_patches_official(seg['data'])
    if patches is not None:
        standardized = standardize_to_target(patches, fixed_channels, median_patches)
        X_incorrect.append(standardized)

X_correct = np.array(X_correct, dtype=np.float32)
X_incorrect = np.array(X_incorrect, dtype=np.float32)

print(f"\nPatches created:")
print(f"  Correct:   {X_correct.shape}")
print(f"  Incorrect: {X_incorrect.shape}")

# Data quality checks
print(f"\nData quality:")
print(f"  Correct mean:   {X_correct.mean():.4f}")
print(f"  Incorrect mean: {X_incorrect.mean():.4f}")

if np.isnan(X_correct).any() or np.isnan(X_incorrect).any():
    print("  ⚠️  WARNING: Data contains NaN!")
    # Remove NaN values
    X_correct = np.nan_to_num(X_correct, nan=0.0)
    X_incorrect = np.nan_to_num(X_incorrect, nan=0.0)
    print("  ✓ NaN values replaced with 0")

if np.isinf(X_correct).any() or np.isinf(X_incorrect).any():
    print("  ⚠️  WARNING: Data contains Inf!")
    # Remove Inf values
    X_correct = np.nan_to_num(X_correct, posinf=0.0, neginf=0.0)
    X_incorrect = np.nan_to_num(X_incorrect, posinf=0.0, neginf=0.0)
    print("  ✓ Inf values replaced with 0")

# Create labels
y_correct = np.ones(len(X_correct), dtype=np.int64)
y_incorrect = np.zeros(len(X_incorrect), dtype=np.int64)

# Combine data
X = np.concatenate([X_correct, X_incorrect], axis=0)
y = np.concatenate([y_correct, y_incorrect], axis=0)

print(f"\n{'=' * 60}")
print(f"COMBINED DATASET")
print(f"{'=' * 60}")
print(f"X: {X.shape} (samples, channels, patches, points)")
print(f"y: {y.shape}")
print(f"Class 0 (Incorrect): {np.sum(y==0)}")
print(f"Class 1 (Correct):   {np.sum(y==1)}")

# Split into train, validation, and test sets
print(f"\n{'=' * 60}")
print(f"TRAIN/VAL/TEST SPLIT")
print(f"{'=' * 60}")

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=42, stratify=y
)

# Second split: separate train and validation
val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
)

print(f"Split ratios: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")
print(f"\nTrain: {X_train.shape[0]} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)}")
print(f"\nValidation: {X_val.shape[0]} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Class 0: {np.sum(y_val==0)}, Class 1: {np.sum(y_val==1)}")
print(f"\nTest: {X_test.shape[0]} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"  Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)}")

# Save data
print(f"\n{'=' * 60}")
print(f"SAVING DATA")
print(f"{'=' * 60}")

np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

# Save configuration
config = {
    'format': 'official_cbramod',
    'n_channels': fixed_channels,
    'n_patches': median_patches,
    'points_per_patch': 200,
    'n_classes': 2,
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test),
    'split_ratios': {
        'train': TRAIN_RATIO,
        'val': VAL_RATIO,
        'test': TEST_RATIO
    },
    'preprocessing': {
        'bandpass': [BANDPASS_LOW, BANDPASS_HIGH],
        'notch': NOTCH_FREQ,
        'target_sfreq': TARGET_SFREQ,
    },
    'channels': KEEP_CHANNELS[:fixed_channels]
}

with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Saved to: {output_dir}")
print(f"\nFiles created:")
print(f"  - X_train.npy: {X_train.shape}")
print(f"  - y_train.npy: {y_train.shape}")
print(f"  - X_val.npy:   {X_val.shape}")
print(f"  - y_val.npy:   {y_val.shape}")
print(f"  - X_test.npy:  {X_test.shape}")
print(f"  - y_test.npy:  {y_test.shape}")
print(f"  - config.json")