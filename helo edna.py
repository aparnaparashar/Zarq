import numpy as np
import pandas as pd
from collections import Counter
from math import log2, exp
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product

# DNA sequencing theory: simulate coverage function based on Lander-Waterman theory
def simulate_coverage(target_length, fragment_length, num_fragments):
    starts = np.random.randint(0, target_length - fragment_length + 1, num_fragments)
    coverage_array = np.zeros(target_length, dtype=bool)
    for start in starts:
        coverage_array[start:start + fragment_length] = True
    coverage_fraction = np.sum(coverage_array) / target_length
    redundancy = (num_fragments * fragment_length) / target_length
    coverage_theory = 1 - exp(-redundancy)  
    expected_contigs = num_fragments * exp(-redundancy)
    return coverage_array, coverage_fraction, coverage_theory, redundancy, expected_contigs

# Model sequencing error probability by position - symmetric parabola
def error_prob_by_position(seq_len):
    center = seq_len / 2
    error_probs = [0.01 + 0.1 * ((i - center) / center) ** 2 for i in range(seq_len)]
    return np.array(error_probs)

# Chimera detection heuristic (naive)
def detect_chimera(sequence, k=6):
    kmers = Counter(sequence[i:i + k] for i in range(len(sequence) - k + 1))
    total = sum(kmers.values())
    max_freq = max(kmers.values()) / total if total > 0 else 0
    return max_freq > 0.2

# Homopolymer detection
def detect_homopolymers(sequence, threshold=5):
    count = 1
    max_count = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 1
    return max_count >= threshold

# GC bias correction - simple lookup/interpolation
def gc_bias_correction(gc_content, observed_abundance, expected_gc_distribution):
    gc_keys = np.array(list(expected_gc_distribution.keys()))
    idx = (np.abs(gc_keys - gc_content)).argmin()
    correction_factor = expected_gc_distribution[gc_keys[idx]]
    corrected_abundance = observed_abundance / correction_factor if correction_factor != 0 else observed_abundance
    return corrected_abundance

# Calculate k-mer frequency vector for cosine similarity
def calculate_kmer_vector(sequence, k=4):
    total_kmers = max(len(sequence) - k + 1, 1)
    kmers = Counter(sequence[i:i + k] for i in range(total_kmers))
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    vector = [kmers.get(kmer, 0) / total_kmers for kmer in all_kmers]
    return np.array(vector).reshape(1, -1)

# Calculate max cosine similarity of query sequence to a list of references
def cosine_similarity_to_references(query_seq, reference_seqs, k=4):
    query_vec = calculate_kmer_vector(query_seq, k)
    max_sim = 0
    for ref_seq in reference_seqs:
        ref_vec = calculate_kmer_vector(ref_seq, k)
        sim = cosine_similarity(query_vec, ref_vec)[0][0]
        if sim > max_sim:
            max_sim = sim
    return max_sim

# Feature extraction function incorporating all features including cosine similarity
def calculate_features(sequence, expected_gc_distribution=None, reference_seqs=None):
    seq_len = len(sequence)
    a = sequence.count('A')
    c = sequence.count('C')
    g = sequence.count('G')
    t = sequence.count('T')
    freqs = {'A': a, 'C': c, 'G': g, 'T': t}

    gc_content = (g + c) / max(seq_len, 1)
    at_content = (a + t) / max(seq_len, 1)
    a_freq = a / max(seq_len, 1)
    c_freq = c / max(seq_len, 1)
    g_freq = g / max(seq_len, 1)
    t_freq = t / max(seq_len, 1)

    gc_skew = (g - c) / max(g + c, 1)
    at_skew = (a - t) / max(a + t, 1)

    entropy = -sum([(freq / seq_len) * log2(freq / seq_len) for freq in freqs.values() if freq > 0])

    k = 2
    kmers = Counter(sequence[i:i + k] for i in range(len(sequence) - k + 1))
    total_kmers = sum(kmers.values())
    kmer_freqs = {f'di_{kmer}': count / total_kmers for kmer, count in kmers.items()}

    chimera_flag = detect_chimera(sequence)
    error_probs = error_prob_by_position(seq_len)
    mean_error_prob = np.mean(error_probs)

    features = {
        'length': seq_len,
        'gc_content': gc_content,
        'at_content': at_content,
        'a_freq': a_freq,
        'c_freq': c_freq,
        'g_freq': g_freq,
        't_freq': t_freq,
        'gc_skew': gc_skew,
        'at_skew': at_skew,
        'entropy': entropy,
        'chimera_flag': int(chimera_flag),
        'mean_positional_error_prob': mean_error_prob,
    }

    features.update(kmer_freqs)

    if expected_gc_distribution is not None:
        features['corrected_gc_content'] = gc_bias_correction(gc_content, gc_content, expected_gc_distribution)
    else:
        features['corrected_gc_content'] = gc_content

    features['sample_shannon_diversity'] = 0.0  # Placeholder - could be expanded
    features['phylo_dist'] = 0.0  # Placeholder

    if reference_seqs is not None:
        features['cosine_sim_max'] = cosine_similarity_to_references(sequence, reference_seqs)
    else:
        features['cosine_sim_max'] = 0.0

    return features

def create_synthetic_dataset(num_samples=1000, seq_len=100):
    bases = ['A', 'C', 'G', 'T']
    sequences = []
    labels = []
    for i in range(num_samples):
        seq = ''.join(np.random.choice(bases, seq_len))
        sequences.append(seq)
        gc_ratio = (seq.count('G') + seq.count('C')) / seq_len
        if gc_ratio < 0.25:
            labels.append('Low_GC')
        elif gc_ratio < 0.5:
            labels.append('Medium_GC')
        else:
            labels.append('High_GC')
    return sequences, labels

def train_model(sequences, labels, expected_gc_distribution=None, reference_seqs=None):
    feature_list = []
    for seq in sequences:
        feats = calculate_features(seq, expected_gc_distribution, reference_seqs)
        feats['has_homopolymer'] = int(detect_homopolymers(seq))
        feature_list.append(feats)
    df_features = pd.DataFrame(feature_list).fillna(0)
    X = df_features.values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)
    model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=100,
                      early_stopping_rounds=10)

    model.save_model('edna_model_independent.txt')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    return model, le

def predict_sequence(model, label_encoder, sequence, expected_gc_distribution=None, reference_seqs=None):
    feats = calculate_features(sequence, expected_gc_distribution, reference_seqs)
    feats['has_homopolymer'] = int(detect_homopolymers(sequence))
    df = pd.DataFrame([feats]).fillna(0)
    preds = model.predict(df.values)
    idx = np.argmax(preds)
    label = label_encoder.inverse_transform([idx])[0]
    confidence = preds[0][idx]
    return label, confidence

if __name__ == "__main__":
    expected_gc_dist = {0.1: 0.9, 0.2: 1.0, 0.3: 1.1, 0.4: 1.05, 0.5: 1.0, 0.6: 0.95}
    reference_seqs = ['ATGCGTAC', 'CGTAGCTA', 'GCTAGCAT']  # Example reference sequences

    print("Generating synthetic dataset...")
    seqs, lbls = create_synthetic_dataset(1000, 100)
    print("Training model...")
    model, le = train_model(seqs, lbls, expected_gc_dist, reference_seqs)

    coverage_array, coverage_frac, coverage_theory, redundancy, expected_contigs = simulate_coverage(10000, 100, 300)
    print(f"Simulated coverage fraction: {coverage_frac:.3f}")
    print(f"Theoretical coverage (Lander-Waterman): {coverage_theory:.3f}")
    print(f"Redundancy: {redundancy:.2f}")
    print(f"Expected number of contigs: {expected_contigs:.2f}")

    test_seq = 'ATGCATGCGTACGTAGCTAGCGTACGACTGATCGTAGCTAGC'
    pred_label, conf = predict_sequence(model, le, test_seq, expected_gc_dist, reference_seqs)
    print(f"Predicted class: {pred_label} with confidence {conf:.4f}")
