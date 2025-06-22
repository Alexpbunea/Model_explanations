import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr
from tensorflow.keras.metrics import AUC
import tensorflow as tf
from deel.influenciae.utils import ORDER

batch_size = 32

def collect_influence(calculator, test_ds, train_ds, k=200):
    d = defaultdict(list)
    for name, calc in calculator.items():                        
        exp_ds = calc.top_k(test_ds.batch(1), train_ds.batch(8), k=k, order=ORDER.DESCENDING)
        for (_, _), scores, _ in exp_ds.as_numpy_iterator():
            d[name].append(scores[0])                            
    return d



"""
LONG TAIL PLOT
"""
def plot_long_tail(scores_list, title):
    ranks = np.arange(1, len(scores_list)+1)
    plt.figure()
    plt.semilogy(ranks, np.sort(scores_list)[::-1])      
    plt.xlabel("Range")
    plt.ylabel("Influence score")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_cdf(scores_list, title):
    sorted_scores = np.sort(scores_list)
    cdf = np.arange(len(sorted_scores)) / float(len(sorted_scores))
    plt.figure()
    plt.plot(sorted_scores, cdf)
    plt.xlabel("Influence score")
    plt.ylabel("Cumulative probability")
    plt.title(f"CDF: {title}")
    plt.grid(True)
    plt.show()

def plot_method_comparison(scores_foi, scores_tracin):
    plt.figure()
    plt.scatter(scores_foi, scores_tracin, alpha=0.5)
    plt.xlabel("FOI influence scores")
    plt.ylabel("TracIn influence scores")
    plt.title("FOI vs TracIn Influence Comparison")
    plt.grid(True)
    plt.plot([min(scores_foi), max(scores_foi)], [min(scores_foi), max(scores_foi)], 'r--')  # diagonal
    plt.show()


"""
RANK SIMILARITY
"""


def rank_similarity(list_a_ids, list_b_ids, top_k=None):
    if top_k is not None:
        list_a_ids = list_a_ids[:top_k]
        list_b_ids = list_b_ids[:top_k]
    
    set_a = set(list_a_ids)
    set_b = set(list_b_ids)
    #print(set_a)
    common_set = set_a & set_b
    union_len = len(set_a | set_b)
    jaccard = len(common_set) / union_len if union_len > 0 else 0.0

    pos_a = {id_: i for i, id_ in enumerate(list_a_ids)}
    pos_b = {id_: i for i, id_ in enumerate(list_b_ids)}
    #print(len(pos_a))
    #print(len(pos_b))
    common = [id_ for id_ in list_a_ids if id_ in set_b]
    #print(common)

    if len(common) < 2:
        tau = np.nan
        spr = np.nan
    else:
        r_a = [pos_a[id_] for id_ in common]
        r_b = [pos_b[id_] for id_ in common]
        tau, _ = kendalltau(r_a, r_b)
        spr, _ = spearmanr(r_a, r_b)
    
    return tau, spr, jaccard


"""
Retrain without k_vals
"""

def retrain_without(original_df, train_features, train_labels, model, test_ds, ids_to_remove):
    # Obtener IDs de entrenamiento originales
    train_ids = original_df.iloc[train_features.index]["PassengerId"].values if hasattr(train_features, 'index') \
                else original_df["PassengerId"][:len(train_features)].values
    
    # Crear máscara para filtrar
    mask = ~np.isin(train_ids, ids_to_remove)
    
    # Filtrar datos
    train_features_filtered = train_features[mask]
    train_labels_filtered = train_labels[mask]
    
    # Crear nuevo dataset
    new_train_ds = tf.data.Dataset.from_tensor_slices(
        (train_features_filtered, train_labels_filtered))
    
    # Clonar y reentrenar modelo
    model_new = tf.keras.models.clone_model(model)
    model_new.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy", AUC(name="auc")])
    model_new.fit(new_train_ds.batch(batch_size), epochs=10, verbose=0)
    
    return model_new.evaluate(test_ds.batch(batch_size), verbose=0)


