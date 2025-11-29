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
import re
import time
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        7000,  # max_active
        200,   # min_active
        17.,   # beam
        8.0,   # lattice_beam
        acoustic_scale,  # acoustic_scale
        1.0,   # ctc_blank_skip_threshold
        0.0,   # length_penalty
        nbest  # nbest
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
    # Remove trailing " |" (with or without spaces) and strip whitespace
    phoneme_sequence = phoneme_sequence.strip()
    if phoneme_sequence.endswith(' |'):
        phoneme_sequence = phoneme_sequence[:-2].strip()
    elif phoneme_sequence.endswith(' | '):
        phoneme_sequence = phoneme_sequence[:-3].strip()

    # Split by " | " to get individual phonemes/groups
    phoneme_groups = phoneme_sequence.split(' | ')

    logits_list = []

    for group in phoneme_groups:
        # Split group into individual phonemes
        phonemes = group.split()

        for phoneme in phonemes:
            phoneme = phoneme.strip()
            if phoneme and phoneme in PHONEME_TO_INDEX:
                # Create logit distribution with high probability for this phoneme
                logits = np.full(41, -float('inf'))  # Start with very low probabilities
                logits[PHONEME_TO_INDEX[phoneme]] = 0.0  # High probability for correct phoneme
                logits[0] = np.log(blank_penalty)  # Blank penalty
                logits_list.append(logits)
            else:
                print(f"Warning: Unknown phoneme '{phoneme}', skipping")
                continue

        # For 1-gram model, we don't add space separators between words
        # The model should handle word boundaries automatically
        pass

    if not logits_list:
        # Return a single blank logit if no valid phonemes
        return np.array([[np.log(blank_penalty)] + [-float('inf')] * 40])

    return np.array(logits_list)


# function for initializing the OPT model and tokenizer
def build_opt(
        model_name='facebook/opt-6.7b',
        cache_dir=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
    
    '''
    Load the OPT-6.7b model and tokenizer from Hugging Face.
    We will load the model with 16-bit precision for faster inference. This requires ~13 GB of VRAM.
    Put the model onto the GPU (if available).
    '''
    
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )

    if device != 'cpu':
        # Move the model to the GPU
        model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # ensure padding token
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# function for rescoring hypotheses with the GPT-2 model
@torch.inference_mode()
def rescore_with_gpt2(
        model,
        tokenizer,
        device,
        hypotheses,
        length_penalty
    ):

    # set model to evaluation mode
    model.eval()

    inputs = tokenizer(hypotheses, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # compute log-probabilities
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    log_probs = log_probs.cpu().numpy()

    input_ids = inputs['input_ids'].cpu().numpy()
    attention_mask = inputs['attention_mask'].cpu().numpy()
    batch_size, seq_len, _ = log_probs.shape

    scores = []
    for i in range(batch_size):
        n_tokens = int(attention_mask[i].sum())
        # sum log-probs of each token given the previous context
        score = sum(
            log_probs[i, t-1, input_ids[i, t]]
            for t in range(1, n_tokens)
        )
        scores.append(score - n_tokens * length_penalty)

    return scores


# function for decoding with the GPT-2 model
def gpt2_lm_decode(
        model,
        tokenizer,
        device,
        nbest,
        acoustic_scale,
        length_penalty,
        alpha,
        returnConfidence=False,
        current_context_str=None,
    ):

    hypotheses = []
    acousticScores = []
    oldLMScores = []

    for out in nbest:

        # get the candidate sentence (hypothesis)
        hyp = out[0].strip()
        if len(hyp) == 0:
            continue

        # add context to the front of each sentence
        if current_context_str is not None and len(current_context_str.split()) > 0:
            hyp = current_context_str + ' ' + hyp
        
        hyp = hyp.replace('>', '')
        hyp = hyp.replace('  ', ' ')
        hyp = hyp.replace(' ,', ',')
        hyp = hyp.replace(' .', '.')
        hyp = hyp.replace(' ?', '?')
        hypotheses.append(hyp)
        acousticScores.append(out[1])
        oldLMScores.append(out[2])

    if len(hypotheses) == 0:
        print('In g2p_lm_decode, len(hypotheses) == 0')
        return ("", []) if not returnConfidence else ("", [], 0.)
    
    # convert to numpy arrays
    acousticScores = np.array(acousticScores)
    oldLMScores = np.array(oldLMScores)

    # get new LM scores from LLM
    try:
        # first, try to rescore all at once
        newLMScores = np.array(rescore_with_gpt2(model, tokenizer, device, hypotheses, length_penalty))

    except Exception as e:
        print(f'Error during OPT rescore: {e}')

        try:
            # if that fails, try to rescore in batches (to avoid VRAM issues)
            newLMScores = []
            for i in range(0, len(hypotheses), int(np.ceil(len(hypotheses)/5))):
                newLMScores.extend(rescore_with_gpt2(model, tokenizer, device, hypotheses[i:i+int(np.ceil(len(hypotheses)/5))], length_penalty))
            newLMScores = np.array(newLMScores)

        except Exception as e:
            print(f'Error during OPT rescore: {e}')
            newLMScores = np.zeros(len(hypotheses))

    # remove context from start of each sentence
    if current_context_str is not None and len(current_context_str.split()) > 0:
        hypotheses = [h[(len(current_context_str)+1):] for h in hypotheses]

    # calculate total scores
    totalScores = (acoustic_scale * acousticScores) + ((1 - alpha) * oldLMScores) + (alpha * newLMScores)

    # get the best hypothesis
    maxIdx = np.argmax(totalScores)
    bestHyp = hypotheses[maxIdx]

    # create nbest output
    nbest_out = []
    min_len = np.min((len(nbest), len(newLMScores), len(totalScores)))
    for i in range(min_len):
        nbest_out.append(';'.join(map(str,[nbest[i][0], nbest[i][1], nbest[i][2], newLMScores[i], totalScores[i]])))

    # return
    if not returnConfidence:
        return bestHyp, nbest_out
    else:
        totalScores = totalScores - np.max(totalScores)
        probs = np.exp(totalScores)
        return bestHyp, nbest_out, probs[maxIdx] / np.sum(probs)


# function to get string differences between two sentences
def get_string_differences(cue, decoder_output):
        decoder_output_words = decoder_output.split()
        cue_words = cue.split()

        @lru_cache(None)
        def reverse_w_backtrace(i, j):
            if i == 0:
                return j, ['I'] * j
            elif j == 0:
                return i, ['D'] * i
            elif i > 0 and j > 0 and decoder_output_words[i-1] == cue_words[j-1]:
                cost, path = reverse_w_backtrace(i-1, j-1)
                return cost, path + [i - 1]
            else:
                insertion_cost, insertion_path = reverse_w_backtrace(i, j-1)
                deletion_cost, deletion_path = reverse_w_backtrace(i-1, j)
                substitution_cost, substitution_path = reverse_w_backtrace(i-1, j-1)
                if insertion_cost <= deletion_cost and insertion_cost <= substitution_cost:
                    return insertion_cost + 1, insertion_path + ['I']
                elif deletion_cost <= insertion_cost and deletion_cost <= substitution_cost:
                    return deletion_cost + 1, deletion_path + ['D']
                else:
                    return substitution_cost + 1, substitution_path + ['R']

        cost, path = reverse_w_backtrace(len(decoder_output_words), len(cue_words))

        # remove insertions from path
        path = [p for p in path if p != 'I']

        # Get the indices in decoder_output of the words that are different from cue
        indices_to_highlight = []
        current_index = 0
        for label, word in zip(path, decoder_output_words):
            if label in ['R','D']:
                indices_to_highlight.append((current_index, current_index+len(word)))
            current_index += len(word) + 1

        return cost, path, indices_to_highlight


def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join(sentence.split())

    return sentence


# function to augment the nbest list by swapping words around, artificially increasing the number of candidates
def augment_nbest(nbest, top_candidates_to_augment=20, acoustic_scale=0.3, score_penalty_percent=0.01):

    sentences = []
    ac_scores = []
    lm_scores = []
    total_scores = []

    for i in range(len(nbest)):
        sentences.append(nbest[i][0].strip())
        ac_scores.append(nbest[i][1])
        lm_scores.append(nbest[i][2])
        total_scores.append(acoustic_scale*nbest[i][1] + nbest[i][2])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # new sentences and scores
    new_sentences = []
    new_ac_scores = []
    new_lm_scores = []
    new_total_scores = []

    # swap words around
    for i1 in range(np.min([len(sentences)-1, top_candidates_to_augment])):
        words1 = sentences[i1].split()

        for i2 in range(i1+1, np.min([len(sentences), top_candidates_to_augment])):
            words2 = sentences[i2].split()

            if len(words1) != len(words2):
                continue
            
            _, path1, _ = get_string_differences(sentences[i1], sentences[i2])
            _, path2, _ = get_string_differences(sentences[i2], sentences[i1])

            replace_indices1 = [i for i, p in enumerate(path2) if p == 'R']
            replace_indices2 = [i for i, p in enumerate(path1) if p == 'R']

            for r1, r2 in zip(replace_indices1, replace_indices2):
                
                new_words1 = words1.copy()
                new_words2 = words2.copy()

                new_words1[r1] = words2[r2]
                new_words2[r2] = words1[r1]

                new_sentence1 = ' '.join(new_words1)
                new_sentence2 = ' '.join(new_words2)

                if new_sentence1 not in sentences and new_sentence1 not in new_sentences:
                    new_sentences.append(new_sentence1)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

                if new_sentence2 not in sentences and new_sentence2 not in new_sentences:
                    new_sentences.append(new_sentence2)
                    new_ac_scores.append(np.mean([ac_scores[i1], ac_scores[i2]]) - score_penalty_percent * np.abs(np.mean([ac_scores[i1], ac_scores[i2]])))
                    new_lm_scores.append(np.mean([lm_scores[i1], lm_scores[i2]]) - score_penalty_percent * np.abs(np.mean([lm_scores[i1], lm_scores[i2]])))
                    new_total_scores.append(acoustic_scale*new_ac_scores[-1] + new_lm_scores[-1])

    # combine new sentences and scores with old
    for i in range(len(new_sentences)):
        sentences.append(new_sentences[i])
        ac_scores.append(new_ac_scores[i])
        lm_scores.append(new_lm_scores[i])
        total_scores.append(new_total_scores[i])

    # sort by total score
    sorted_indices = np.argsort(total_scores)[::-1]
    sentences = [sentences[i] for i in sorted_indices]
    ac_scores = [ac_scores[i] for i in sorted_indices]
    lm_scores = [lm_scores[i] for i in sorted_indices]
    total_scores = [total_scores[i] for i in sorted_indices]

    # return nbest
    nbest_out = []
    for i in range(len(sentences)):
        nbest_out.append([sentences[i], ac_scores[i], lm_scores[i]])

    return nbest_out


def predict_words_from_phonemes(csv_path, lm_path, output_path=None,
                               acoustic_scale=0.325, blank_penalty=90.0, nbest=100,
                               do_opt=False, alpha=0.5, gpu_number=0, opt_cache_dir=None,
                               top_candidates_to_augment=20, score_penalty_percent=0.01):
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

    # Initialize OPT model if needed
    lm = None
    lm_tokenizer = None
    device = None
    if do_opt:
        print(f"Initializing OPT model...")
        device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        lm, lm_tokenizer = build_opt(cache_dir=opt_cache_dir, device=device)
        print("OPT model initialized.")

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
            # Reset decoder for clean state
            decoder.Reset()

            # Convert phoneme sequence to logits
            logits = phoneme_sequence_to_logits(phoneme_sequence, blank_penalty=blank_penalty)

            # Run language model decoding (logits should be 2D: T x 41)
            lm_decoder.DecodeNumpy(decoder, logits, np.zeros_like(logits),
                                  np.log(blank_penalty))

            # Finalize decoding
            decoder.FinishDecoding()

            # Get results
            lm_results = decoder.result()
            
            best_prediction = ""
            acoustic_score = 0.0
            lm_score = 0.0
            llm_score = 0.0
            total_score = -float('inf')

            if len(lm_results) > 0:
                # Prepare nbest list for augmentation/rescoring
                nbest_out = []
                for d in lm_results:
                    nbest_out.append([d.sentence, d.ac_score, d.lm_score])
                
                # Augment nbest if needed
                if nbest > 1:
                    nbest_out = augment_nbest(
                        nbest=nbest_out,
                        top_candidates_to_augment=top_candidates_to_augment,
                        acoustic_scale=acoustic_scale,
                        score_penalty_percent=score_penalty_percent
                    )

                if do_opt:
                    best_prediction, nbest_redis, confidences = gpt2_lm_decode(
                        lm,
                        lm_tokenizer,
                        device,
                        nbest_out,
                        acoustic_scale,
                        alpha=alpha,
                        length_penalty=0.0,
                        returnConfidence=True
                    )
                    # Parse best prediction scores from nbest_redis if possible, or just use what we have
                    # The gpt2_lm_decode returns bestHyp string.
                    # We can try to find it in nbest_redis to get scores
                    
                    # nbest_redis is a list of strings: "sentence;ac_score;lm_score;llm_score;total_score"
                    for item in nbest_redis:
                        parts = item.split(';')
                        if parts[0] == best_prediction:
                            acoustic_score = float(parts[1])
                            lm_score = float(parts[2])
                            llm_score = float(parts[3])
                            total_score = float(parts[4])
                            break
                else:
                    # Just take the best from nbest_out
                    best_prediction = nbest_out[0][0]
                    acoustic_score = nbest_out[0][1]
                    lm_score = nbest_out[0][2]
                    total_score = acoustic_scale * acoustic_score + lm_score

            else:
                best_prediction = ""
                acoustic_score = 0.0
                lm_score = 0.0
                total_score = 0.0

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            best_prediction = ""
            acoustic_score = 0.0
            lm_score = 0.0
            total_score = 0.0
            llm_score = 0.0

        results.append({
            'session': session,
            'block': block,
            'trial': trial,
            'pred_phonemes': phoneme_sequence,
            'pred_words': best_prediction,
            'acoustic_score': acoustic_score,
            'lm_score': lm_score,
            'llm_score': llm_score,
            'total_score': total_score
        })

    # Save results
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"word_predictions_{timestamp}.csv"

    # Write results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['session', 'block', 'trial', 'pred_phonemes', 'pred_words',
                     'acoustic_score', 'lm_score', 'llm_score', 'total_score']
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
        print(f"Average Total Score: {avg_score:.2f}")

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
    
    # New arguments for OPT
    parser.add_argument('--do_opt', action='store_true',
                       help='Enable OPT rescoring')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Alpha parameter for mixing OPT scores')
    parser.add_argument('--gpu_number', type=int, default=0,
                       help='GPU number to use')
    parser.add_argument('--opt_cache_dir', type=str, default=None,
                       help='Cache directory for OPT model')
    parser.add_argument('--top_candidates_to_augment', type=int, default=20,
                       help='Number of top candidates to augment')
    parser.add_argument('--score_penalty_percent', type=float, default=0.01,
                       help='Score penalty percent for augmentation')

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
        nbest=args.nbest,
        do_opt=args.do_opt,
        alpha=args.alpha,
        gpu_number=args.gpu_number,
        opt_cache_dir=args.opt_cache_dir,
        top_candidates_to_augment=args.top_candidates_to_augment,
        score_penalty_percent=args.score_penalty_percent
    )


if __name__ == "__main__":
    main()
