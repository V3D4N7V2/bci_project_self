#!/usr/bin/env python3
"""
Script to predict words from phoneme sequences using the 1-gram language model.
Reads predicted phonemes from a CSV file and outputs word predictions.
"""

import os
import sys
import argparse
import csv
import numpy as np
import lm_decoder
import torch
from datetime import datetime

# Phoneme mapping from evaluate_model_helpers.py
LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',  # space separator
]

# Create reverse mapping
PHONEME_TO_INDEX = {phoneme: idx for idx, phoneme in enumerate(LOGIT_TO_PHONEME)}


def build_lm_decoder(model_path, acoustic_scale=0.325, blank_penalty=90.0, nbest=100):
    """
    Initialize the ngram language model decoder.
    """
    decode_opts = lm_decoder.DecodeOptions(
        max_active=7000,
        min_active=200,
        beam=17.,
        lattice_beam=8.0,
        acoustic_scale=acoustic_scale,
        ctc_blank_skip_threshold=1.0,
        length_penalty=0.0,
        nbest=nbest
    )

    TLG_path = os.path.join(model_path, 'TLG.fst')
    words_path = os.path.join(model_path, 'words.txt')
    G_path = os.path.join(model_path, 'G.fst')
    rescore_G_path = os.path.join(model_path, 'G_no_prune.fst')

    if not os.path.exists(rescore_G_path):
        rescore_G_path = ""
        G_path = ""
    if not os.path.exists(TLG_path):
        raise ValueError(f'TLG file not found at {TLG_path}')
    if not os.path.exists(words_path):
        raise ValueError(f'words file not found at {words_path}')

    decode_resource = lm_decoder.DecodeResource(
        TLG_path,
        G_path,
        rescore_G_path,
        words_path,
        ""
    )
    decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)

    return decoder


def phoneme_sequence_to_logits(phoneme_sequence, blank_penalty=90.0):
    """
    Convert a phoneme sequence string to logits array.

    Args:
        phoneme_sequence: String like "Y UW  |  K AE N  |  S IY"
        blank_penalty: Penalty for blank tokens

    Returns:
        numpy array of shape (T, 41) where T is sequence length
    """
    # Split by " | " to get individual phonemes/groups
    phoneme_groups = phoneme_sequence.strip().split(' | ')

    logits_list = []

    for group in phoneme_groups:
        if not group.strip():
            continue

        # Split group into individual phonemes
        phonemes = group.strip().split()

        for phoneme in phonemes:
            phoneme = phoneme.strip()
            if phoneme in PHONEME_TO_INDEX:
                # Create logit distribution with high probability for this phoneme
                logits = np.full(41, -float('inf'))  # Start with very low probabilities
                logits[PHONEME_TO_INDEX[phoneme]] = 0.0  # High probability for correct phoneme
                logits[0] = np.log(blank_penalty)  # Blank penalty
                logits_list.append(logits)
            else:
                print(f"Warning: Unknown phoneme '{phoneme}', skipping")
                continue

        # Add space separator after each group (except possibly the last)
        if group != phoneme_groups[-1]:
            logits = np.full(41, -float('inf'))
            logits[PHONEME_TO_INDEX[' | ']] = 0.0
            logits[0] = np.log(blank_penalty)
            logits_list.append(logits)

    if not logits_list:
        # Return a single blank logit if no valid phonemes
        return np.array([[np.log(blank_penalty)] + [-float('inf')] * 40])

    return np.array(logits_list)


def predict_words_from_phonemes(csv_path, lm_path, output_path=None,
                               acoustic_scale=0.325, blank_penalty=90.0, nbest=100):
    """
    Read phoneme predictions from CSV and generate word predictions using language model.
    """
    print(f"Loading data from {csv_path}")

    # Read CSV using built-in csv module
    data = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    if not data:
        raise ValueError(f"No data found in CSV file: {csv_path}")

    # Check if required columns exist
    if 'pred_phonemes' not in data[0]:
        raise ValueError("CSV must contain 'pred_phonemes' column")

    print(f"Initializing language model from {lm_path}")
    decoder = build_lm_decoder(lm_path, acoustic_scale=acoustic_scale,
                              blank_penalty=blank_penalty, nbest=nbest)

    results = []

    print(f"Processing {len(data)} samples...")

    for idx, row in enumerate(data):
        if idx % 10 == 0:
            print(f"Processing sample {idx+1}/{len(data)}")

        session = row.get('session', f'sample_{idx}')
        block = row.get('block', '0')
        trial = row.get('trial', str(idx))

        phoneme_sequence = row['pred_phonemes']

        try:
            # Convert phoneme sequence to logits
            logits = phoneme_sequence_to_logits(phoneme_sequence, blank_penalty=blank_penalty)

            # Reshape for language model (add batch dimension)
            logits = logits.reshape(1, -1, 41)  # (1, T, 41)

            # Run language model decoding
            lm_decoder.DecodeNumpy(decoder, logits[0], np.zeros_like(logits[0]),
                                  np.log(blank_penalty))

            # Finalize decoding
            decoder.FinishDecoding()

            # Get results
            lm_results = decoder.result()

            if len(lm_results) > 0:
                best_prediction = lm_results[0].sentence
                acoustic_score = lm_results[0].ac_score
                lm_score = lm_results[0].lm_score
            else:
                best_prediction = ""
                acoustic_score = 0.0
                lm_score = 0.0

            # Reset decoder for next sample
            decoder.Reset()

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            best_prediction = ""
            acoustic_score = 0.0
            lm_score = 0.0

        results.append({
            'session': session,
            'block': block,
            'trial': trial,
            'pred_phonemes': phoneme_sequence,
            'pred_words': best_prediction,
            'acoustic_score': acoustic_score,
            'lm_score': lm_score,
            'total_score': acoustic_scale * acoustic_score + lm_score
        })

    # Save results
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"word_predictions_{timestamp}.csv"

    # Write results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['session', 'block', 'trial', 'pred_phonemes', 'pred_words',
                     'acoustic_score', 'lm_score', 'total_score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")

    # Print summary
    valid_predictions = [r for r in results if len(r['pred_words']) > 0]
    print("Summary:")
    print(f"Total samples: {len(results)}")
    print(f"Valid predictions: {len(valid_predictions)}")
    if len(results) > 0:
        avg_score = sum(r['total_score'] for r in valid_predictions) / len(valid_predictions) if valid_predictions else 0
        print(".2f")

    return results


def main():
    parser = argparse.ArgumentParser(description='Predict words from phoneme sequences using language model')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with predicted phonemes')
    parser.add_argument('--lm_path', type=str,
                       default='language_model/pretrained_language_models/openwebtext_1gram_lm_sil',
                       help='Path to language model directory')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save word predictions (default: auto-generated)')
    parser.add_argument('--acoustic_scale', type=float, default=0.325,
                       help='Acoustic scale parameter')
    parser.add_argument('--blank_penalty', type=float, default=90.0,
                       help='Blank penalty parameter')
    parser.add_argument('--nbest', type=int, default=100,
                       help='Number of best candidates to consider')

    args = parser.parse_args()

    # Check if we're in the right conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if 'b2txt25_lm' not in conda_env:
        print("Warning: Not in b2txt25_lm conda environment. Please run:")
        print("conda activate b2txt25_lm")
        print("Continuing anyway...")

    # Convert relative paths to absolute
    args.csv_path = os.path.abspath(args.csv_path)
    args.lm_path = os.path.abspath(args.lm_path)
    if args.output_path:
        args.output_path = os.path.abspath(args.output_path)

    # Check paths exist
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    if not os.path.exists(args.lm_path):
        raise FileNotFoundError(f"Language model path not found: {args.lm_path}")

    # Run prediction
    predict_words_from_phonemes(
        csv_path=args.csv_path,
        lm_path=args.lm_path,
        output_path=args.output_path,
        acoustic_scale=args.acoustic_scale,
        blank_penalty=args.blank_penalty,
        nbest=args.nbest
    )


if __name__ == "__main__":
    main()
