import os
# import pickle5 as pickle
import pickle
import numpy as np
import pandas as pd
from scipy.special import comb


# from lib import BoB, bob_from_bitarray
from collections import Counter, defaultdict

np.random.seed(3)

MIN_SITE_COUNT = 30
# TOPK = 1

def load_results(results_in, sim_in = None):
    with open(results_in, 'rb') as handle:
        bobs_raw = pickle.load(handle)
    slides, bob, mosaics, sjids = list(zip(*bobs_raw))
    slides = np.array([os.path.basename(s).split('.')[0] for s in slides])

    indices = []
    for idx in range(len(slides)):
        if bob[idx] is None:
            indices.append(False)
        else:
            indices.append(True)

    slides = np.array(slides)[indices]
    bob = np.array(bob, dtype = object)[indices]
    mosaics = np.array(mosaics, dtype = object)[indices]
    sjids = np.array(sjids, dtype = object)[indices]

    bobs = [bob_from_bitarray(b) for b in bob]
    N = len(bobs)

    #Load sim matrix
    if sim_in is not None:
        with open(sim_in, 'rb') as handle:
            similarity_matrix = pickle.load(handle)
    else:
        similarity_matrix = None

    return slides, bobs, mosaics, sjids, similarity_matrix

def get_class(scan_id, all_metadata):
    return all_metadata.loc[all_metadata['Slide.Scan.File'] == scan_id, 'Tumor.group'].values[0]

def make_weighted_new_row(slide, query_class, pred, generous, nns, normalized_nns, correct_percentage):
    return pd.DataFrame({'query': [slide], 
                        'query_class': [query_class], 
                        'prediction': [pred],
                        'generous_tie': [generous],
                        'nearest_neighbors': [nns],
                        'normalized_nearest_neighbors': [normalized_nns],
                        'correct_percentage': [correct_percentage]})


def make_new_row(slide, query_class, pred, generous, nns):
    return pd.DataFrame({'query': [slide], 
                        'query_class': [query_class], 
                        'prediction': [pred],
                        'generous_tie': [generous],
                        'nearest_neighbors': [nns]})

def load_results_new(sim_in):
    try:
        sim = pd.read_pickle(sim_in)
        sim.index = sim.index.astype(str)
        sim.columns = sim.columns.astype(str)
        print(sim.index)
    except:
        sim = pd.read_csv(sim_in, index_col = 0)
        print(sim)
        sim.index = sim.index.astype(str)
        sim.columns = sim.columns.astype(str)
        print(sim.index)
    return sim

def load_metadata(metadata_path, min_site_count = MIN_SITE_COUNT, disease_name = 'Tumor.group'):
    all_metadata = pd.read_csv(metadata_path)
    all_metadata = all_metadata.dropna(subset=['Slide.Scan.File'])
    all_metadata[disease_name] = all_metadata[disease_name].apply(lambda x: str(x).upper().replace(' ', '_'))
    populated_sites = all_metadata[disease_name].value_counts()
    populated_sites = list(populated_sites[populated_sites >= min_site_count].index)
    all_metadata = all_metadata[all_metadata[disease_name].isin(populated_sites)]
    all_metadata['Slide.Scan.File'] = all_metadata['Slide.Scan.File'].astype(int).astype(str)
    print(f"Size of metadata: {len(all_metadata)}")
    return all_metadata

def filter_data_new(sim, metadata, REMOVE_NORMAL = False):
    print(len(sim))
    slides = sim.index
    # print(type(slides[0]))
    print("Slides before filtering", len(slides))

    slides = slides[slides != None]
    slides = slides[np.isin(slides, metadata['Slide.Scan.File'])]
    
    # print("Slides after filtering", len(slides))

    # print(metadata['Slide.Scan.File'])

    sim_filtered = sim.loc[slides, slides]
    metadata = metadata[metadata['Slide.Scan.File'].isin(slides)].drop_duplicates(subset=['Slide.Scan.File'])
    # Create a list of Slide.Scan.File where Tumor.group is "MISC"
    misc_slides = metadata.loc[metadata['Tumor.group'] == "MISC", 'Slide.Scan.File'].tolist()
    print(f"Number of misc slides: {len(misc_slides)}")
    # Remove rows in metadata where Tumor.group is "MISC"
    metadata = metadata[metadata['Tumor.group'] != "MISC"]

    # Remove rows and columns in sim_filtered corresponding to misc_slides
    sim_filtered = sim_filtered.drop(index=misc_slides, columns=misc_slides, errors='ignore')

    # Commented out to keep all slides for now
    # slides_to_remove = ['15301', '10111', '10113', '12886']
    # sim_filtered = sim_filtered.drop(index=slides_to_remove, columns=slides_to_remove, errors='ignore')
    
    if REMOVE_NORMAL == True:
        normal_slides = metadata.loc[metadata['Tumor.group'] == "NORMAL", 'Slide.Scan.File'].tolist()
        print(f"Number of normal slides: {len(normal_slides)}")
        metadata = metadata[metadata['Tumor.group'] != "NORMAL"]
        # Remove rows and columns in sim_filtered corresponding to normal_slides
        sim_filtered = sim_filtered.drop(index=normal_slides, columns=normal_slides, errors='ignore')

    
    # Remove rows and columns with all NaN values
    sim_filtered = sim_filtered.dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    # print(f"Size of metadata after filtering: {len(metadata)}")

    slides = sim_filtered.index
    print("Slides after filtering", len(slides))
    return sim_filtered, slides, metadata

#Filter results based on this table
def filter_data(slides, bob, mosaics, sjids, similarity_matrix, all_metadata):
    indices = []
    for idx in range(len(slides)):
        indices.append(slides[idx] in list(all_metadata['query']))

    slides = slides[indices]
    bob = np.array(bob, dtype = object)[indices]
    mosaics = mosaics[indices]
    sjids = sjids[indices]
    similarity_matrix = similarity_matrix[:,indices][indices,:]
    return slides, bob, mosaics, sjids, similarity_matrix, indices

## Find closest
def make_nn_predictions_tie_breaker_generous(all_metadata, slides, similarity_matrix, topk):
    N = len(similarity_matrix)
    results = pd.DataFrame(columns = ['query', 'query_class', 'prediction', 'generous_tie', 'nearest_neighbors'])
    for idx in range(N):
        slide = slides[idx]
        query_class = get_class(slide, all_metadata)
        closest_indices = np.argsort(similarity_matrix.iloc[idx,:].values)[:topk]
        closest_slides = [slides[i] for i in closest_indices]
        closest_classes = [get_class(s, all_metadata) for s in closest_slides]
        closest_similarities = [similarity_matrix.iloc[idx, i] for i in closest_indices]
        counts = Counter(closest_classes)
        max_count = max([c for c in counts.values()])
        preds = [k for k,v in counts.items() if v == max_count]
        pred_counts = [(k,v) for k,v in counts.items() if v == max_count]
        # print(pred_counts)
        # nns = pred_counts
        # Initialize a defaultdict
        classes_similarities_dict = defaultdict(list)

        # Assuming closest_classes and closest_similarities are already defined
        # for key, value in zip(closest_classes, closest_similarities):
            # classes_similarities_dict[key].append(value)
        for slide_id, tumor_type, sim in zip(closest_slides, closest_classes, closest_similarities):
            classes_similarities_dict[tumor_type].append((slide_id, sim))
        nns = classes_similarities_dict

        # if query_class in closest_classes:
            # pred = query_class
            # generous = 1

        if query_class in preds:
            pred = query_class
            generous = 1

        else:
            generous = 0
            # pred = np.random.choice(preds)
            pred = preds[0]


        results = pd.concat([results, make_new_row(slide, query_class, pred, generous, nns)])
    return results

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def make_nn_predictions_tie_breaker_binary(all_metadata, slides, similarity_matrix, topk):
    """
    Very lenient evaluator.
    binary_hit = 1 if the true class appears at least once among the top-k neighbors
    binary_hit = 0 otherwise

    prediction field:
      if binary_hit == 1, set prediction to the true class (since it is present in top-k)
      else fall back to a simple tie-breaker among the most frequent classes in top-k
           using the first encountered tied class for determinism
    """
    N = len(similarity_matrix)
    results = pd.DataFrame(columns=['query', 'query_class', 'prediction', 'binary_hit', 'nearest_neighbors'])

    for idx in range(N):
        slide = slides[idx]
        query_class = get_class(slide, all_metadata)

        # Use argsort (not argpartition) so top_{k+1} is always a superset of top_k
        closest_indices = np.argsort(similarity_matrix.iloc[idx, :].values)[:topk]

        # Gather neighbor info
        closest_slides = [slides[i] for i in closest_indices]
        closest_classes = [get_class(s, all_metadata) for s in closest_slides]
        closest_similarities = [similarity_matrix.iloc[idx, i] for i in closest_indices]

        # Map each class to list of (neighbor_slide_id, similarity)
        classes_similarities_dict = defaultdict(list)
        for slide_id, tumor_type, sim in zip(closest_slides, closest_classes, closest_similarities):
            classes_similarities_dict[tumor_type].append((slide_id, sim))
        nns = classes_similarities_dict

        # Binary success if the true class appears anywhere in top-k
        binary_hit = 1 if query_class in closest_classes else 0

        # Choose a prediction to record
        if binary_hit == 1:
            # Since the true class is present, record it as the prediction
            pred = query_class
        else:
            # Fall back to a simple argmax over counts, deterministic tie break
            counts = Counter(closest_classes)
            max_count = max(counts.values())
            tied = [k for k, v in counts.items() if v == max_count]
            pred = tied[0]

        # Append row
        results = pd.concat([results, make_new_row(slide, query_class, pred, binary_hit, nns)], ignore_index=True)

    return results


import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def make_nn_predictions_tie_breaker_absolute(all_metadata, slides, similarity_matrix, topk=5):
    """
    Absolute-majority rule on Top-k (distance-like matrix: smaller = closer).
    Success (absolute_majority) = 1 iff TRUE class count >= (floor(k/2) + 1).
    prediction = the unique class with >= need; otherwise 'Ambiguous'.
    NOTE: This mirrors your current neighbor picking (argpartition on the row,
    no self-exclusion, no ordering of the Top-k).
    """
    N = len(similarity_matrix)
    results = pd.DataFrame(columns=['query', 'query_class', 'prediction', 'absolute_majority', 'nearest_neighbors'])
    need = (topk // 2) + 1  # > half; for k=5 this is 3

    for idx in range(N):
        slide = slides[idx]
        query_class = get_class(slide, all_metadata)

        row = similarity_matrix.iloc[idx, :].to_numpy()
        k = min(topk, len(row))
        closest_indices = np.argsort(row)[:k]

        closest_slides = [slides[i] for i in closest_indices]
        closest_classes = [get_class(s, all_metadata) for s in closest_slides]
        closest_similarities = [similarity_matrix.iloc[idx, i] for i in closest_indices]

        # Count hits
        counts = Counter(closest_classes)
        hits_true = counts.get(query_class, 0)
        absolute = int(hits_true >= need)

        # Prediction: unique class with >= need; else Ambiguous
        maj_classes = [c for c, v in counts.items() if v >= need]
        pred = maj_classes[0] if len(maj_classes) == 1 else 'Ambiguous'

        # Build NN dict: class -> [(slide_id, sim), ...]
        classes_similarities_dict = defaultdict(list)
        for slide_id, tumor_type, sim in zip(closest_slides, closest_classes, closest_similarities):
            classes_similarities_dict[tumor_type].append((slide_id, sim))

        # If you use make_new_row(slide, query_class, pred, metric_value, nns), swap this in:
        results = pd.concat([results, make_new_row(slide, query_class, pred, absolute, classes_similarities_dict)])


    return results
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def make_nn_predictions_tie_breaker_closest(all_metadata, slides, similarity_matrix, topk):
    """
    MV@k with deterministic tie-break:
      - Compute Top-k neighbors (smaller distance = closer).
      - If there's a unique plurality/majority, predict that class.
      - If there's a tie for max count, pick the class of the SINGLE CLOSEST neighbor
        among the tied classes (scan the Top-k from closest to farthest).
    Returns a DataFrame with the SAME columns as your generous function:
      ['query','query_class','prediction','generous_tie','nearest_neighbors']
    Here, 'generous_tie' is used as a correctness flag: 1 if prediction == query_class else 0.
    """
    N = len(similarity_matrix)
    results = pd.DataFrame(columns=['query', 'query_class', 'prediction', 'generous_tie', 'nearest_neighbors'])

    for idx in range(N):
        slide = slides[idx]
        query_class = get_class(slide, all_metadata)

        row = similarity_matrix.iloc[idx, :].to_numpy()
        k = min(topk, len(row))
        closest_indices = np.argsort(row)[:k]

        # Build Top-k lists
        top_slides = [slides[i] for i in closest_indices]
        top_classes = [get_class(s, all_metadata) for s in top_slides]
        top_dists   = [row[i] for i in closest_indices]

        # Count per class over Top-k
        counts = Counter(top_classes)
        max_count = max(counts.values())
        tied_max = [c for c, v in counts.items() if v == max_count]

        # Majority / plurality with tie-break-by-closest
        if len(tied_max) == 1:
            pred = tied_max[0]
        else:
            # Scan from the closest neighbor upward; pick the first whose class is in tied_max
            pred = None
            for cls in top_classes:  # already ordered by closeness
                if cls in tied_max:
                    pred = cls
                    break
            # Safety fallback (shouldn't happen): if none matched, pick the first class
            if pred is None:
                pred = top_classes[0]

        # Correctness flag (reuse your column name for compatibility)
        correct_flag = int(pred == query_class)

        # NN dictionary: class -> list[(slide_id, distance)]
        classes_similarities_dict = defaultdict(list)
        for s_id, cls, d in zip(top_slides, top_classes, top_dists):
            classes_similarities_dict[cls].append((s_id, d))

        # If you have a helper make_new_row(...), keep using it:
        # results = pd.concat([results, make_new_row(slide, query_class, pred, correct_flag, classes_similarities_dict)])
        results = pd.concat([results, pd.DataFrame([{
            'query': slide,
            'query_class': query_class,
            'prediction': pred,
            'generous_tie': correct_flag,          # correctness flag for this MV@k run
            'nearest_neighbors': classes_similarities_dict
        }])], ignore_index=True)

    return results


def make_nn_predictions(all_metadata, slides, similarity_matrix):
    N = len(slides)
    results = pd.DataFrame({'query':[], 'hit_slide':[], 'query_class':[], 'hit_class':[]})

    for idx in range(N):
        slide = slides[idx].replace('_', '')
        qc = all_metadata.loc[all_metadata['query'] == slide, 'diagnosis'].values
        if not len(qc): continue
        qc = qc[0]
        closest_indices = np.argpartition(similarity_matrix[idx,:],TOPK+1)[:(TOPK+1)]
        closest_slides = [slides[i].replace('_', '') for i in closest_indices]
        if slide in closest_slides: closest_slides.remove(slide)
        for cs in closest_slides:
            hc = all_metadata.loc[all_metadata['query'] == cs, 'diagnosis'].values
            if not len(hc): continue
            hc = hc[0]
            newrow = pd.DataFrame({'query': [slide], 'hit_slide': [cs], 'query_class': [qc], 'hit_class': [hc]})
            results = pd.concat([results, newrow])
            
    results = results.reset_index(drop = True)

    results_summary = results.groupby('query')['hit_class'].agg(lambda x: list(pd.Series.mode(x))).reset_index()
    results_summary = pd.merge(results_summary, all_metadata, on = 'query', how = 'left')
    results_summary = results_summary.rename(columns = {'hit_class': 'prediction', 'diagnosis':'label'})

    #Generous tie breaking
    count = 0
    gen_count = 0
    results_summary['tied'] = False
    for idx in range(len(results_summary)):
        pred = results_summary.loc[idx, 'prediction']
        lab = results_summary.loc[idx,'label']
        if type(pred) in [np.ndarray, list]:
            count += 1
            if len(pred) > 1:
                results_summary.loc[idx, 'tied'] = True
            if lab in pred:
                results_summary.loc[idx,'prediction'] = lab
                gen_count += 1
            else:
                results_summary.loc[idx,'prediction'] = pred[np.random.randint(0, 1)]

    results_summary = results_summary[results_summary['tied'] == False]

    diag_counts = [(k,v['diagnosis']) for k,v in all_metadata.diagnosis.value_counts().reset_index().set_index('index').to_dict('index').items()]
    diag_counts = sorted(diag_counts, key = lambda x: x[1], reverse = True)
    diag_sorted = [d[0] for d in diag_counts]

    print("Number of ties", count)
    print("Number of generous tie breaks", gen_count)
    return results_summary, diag_sorted


def get_prob_table(metadata):

    populated_sites = metadata['Tumor.group'].value_counts().to_dict()
    total_images_num = len(metadata)
    probabilities_random = {cat: count / total_images_num for cat, count in populated_sites.items()}

    def binomial_probability(n, k, p):
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    probs = {cat: [binomial_probability(TOPK, k, prob) for k in range(1, TOPK+1)] for cat, prob in probabilities_random.items()}

    probs_df = pd.DataFrame(probs).transpose()
    probs_df.columns = np.linspace(1, TOPK, TOPK, dtype=int)

    return probs_df

def find_percentile(value, data):
    total_counts = sum(data.values())  
    counts_greater_than_value = sum(frequency for bit_count, frequency in data.items() if int(bit_count) >= value)
    percentile = 100 - (counts_greater_than_value / total_counts) * 100
    return percentile

def normalize_reciprocal_values(d):
    # Calculate the reciprocal of each value (1/value)
    reciprocals = {k: 1/v for k, v in d.items()}
    
    # Sum the reciprocals to use as the normalization denominator
    total = sum(reciprocals.values())
    
    # Normalize the reciprocals and return the new dictionary
    return {k: v / total for k, v in reciprocals.items()}


## Find closest
def make_weighted_nn_predictions(all_metadata, slides, similarity_matrix, baseline, topk):

    N = len(similarity_matrix)
    results = pd.DataFrame(columns = ['query', 'query_class', 'prediction', 'generous_tie', 'nearest_neighbors', 'normalized_nearest_neighbors', 'correct_percentage'])
    probabilities_exact_picks_df = get_prob_table(all_metadata)
    for idx in range(N):
        slide = slides[idx]
        query_class = get_class(slide, all_metadata)
        closest_indices = np.argsort(similarity_matrix.iloc[idx,:].values)[:topk]
        closest_similarities = [similarity_matrix.iloc[idx, i] for i in closest_indices]
        closest_slides = [slides[i] for i in closest_indices]
        closest_classes = [get_class(s, all_metadata) for s in closest_slides]
        category_counts = Counter(closest_classes)

        weighted_scores = {}
        for cat, count in category_counts.items():
            prob_for_count = probabilities_exact_picks_df.loc[cat, count]
            # avg_score = sum(cat_score for cat_score, cat_name in zip(closest_similarities, closest_classes) if cat_name == cat) / count
            # avg_score_percentile = find_percentile(avg_score, baseline) 
            # weighted_score = avg_score_percentile * prob_for_count 
            min_score = min(cat_score for cat_score, cat_name in zip(closest_similarities, closest_classes) if cat_name == cat)
            min_score_percentile = find_percentile(min_score, baseline) 
            weighted_score = min_score_percentile * prob_for_count 

            weighted_scores[cat] = weighted_score
        
        pred = min(weighted_scores, key=weighted_scores.get)

         # Normalize and choose based on probability
        # inverted_scores = {cat: 1.0 / score for cat, score in weighted_scores.items()}
        # total_inverted_score = sum(inverted_scores.values())
        # normalized_scores = {cat: score / total_inverted_score for cat, score in inverted_scores.items()}
        
        # pred = np.random.choice(list(normalized_scores.keys()), p=list(normalized_scores.values()))
               
        
        if pred == query_class:
            generous = 1
        else:
            generous = 0

        nns = weighted_scores
        normalized_nns = normalize_reciprocal_values(nns)
        
        correct_percentage = normalized_nns.get(query_class, 0)
        results = pd.concat([results, make_new_row(slide, query_class, pred, generous, nns, normalized_nns, correct_percentage)])

    return results