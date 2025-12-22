import time
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from typing import List, Tuple

def load_and_scale_data(train_path: str, 
                        val_path: str, test_df: pd.DataFrame)-> Tuple[np.ndarray,
                                                                    np.ndarray,
                                                                    np.ndarray,
                                                                    np.ndarray, 
                                                                    np.ndarray]:
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    X_train = train_df.drop('class', axis=1)
    y_train = train_df['class'].values

    X_val = val_df.drop('class', axis=1)
    y_val = val_df['class'].values

    X_test = test_df.values

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)


    return X_train, X_val, X_test, y_train, y_val


def nnr_predict_weighted(distances, y_train, radius) -> List:

    predictions = []

    #every column is all the distances from one test point to all train points
    for i in range(distances.shape[1]):
        
        d = distances[:, i]

        #binary series of neighbors within radius
        neighbor_mask = d <= radius
        neighbor_dists = d[neighbor_mask]
        neighbor_classes = y_train[neighbor_mask]
        
        if len(neighbor_classes) > 0:
            # Weight by inverse distance (+1e-10 to avoid division by zero)
            weights = 1.0 / (neighbor_dists + 1e-10)

            weighted_votes = defaultdict(float)

            for cls, w in zip(neighbor_classes, weights):
                weighted_votes[cls] += w

             #find the key with the max votes not the max value   
            pred = max(weighted_votes, key=weighted_votes.get)

        else:
             # Use 3 nearest neighbors as fallback
            k_nearest_idx = np.argsort(d)[:3]
            pred = Counter(y_train[k_nearest_idx]).most_common(1)[0][0]

        
        predictions.append(pred)
    
    return predictions

def find_best_radius(train_val_distances,
                     y_train,
                     y_val,
                     radii) -> float:
    
    best_radius = radii[0]
    best_acc = 0.0

    for r in radii:
        preds = nnr_predict_weighted(train_val_distances, y_train, r)
        acc = accuracy_score(y_val, preds)

        if acc > best_acc:
            best_acc = acc
            best_radius = r

    return best_radius



def classify_with_NNR(data_trn: str,
                      data_vld: str,
                      df_tst: pd.DataFrame) -> List:

    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    # Load and scale data
    X_train, X_val, X_test, y_train, y_val = load_and_scale_data(data_trn, data_vld, df_tst)

    # Compute array of distances between train-val and train-test points
    train_val_dist = cdist(X_train, X_val, metric ='cityblock') 
    train_test_dist = cdist(X_train, X_test, metric='cityblock')

    all_distances = np.concatenate([train_val_dist.flatten(), train_test_dist.flatten()])

    # Radius search range
    prencentiles = np.linspace(5,95,60)
    radii = np.percentile(all_distances, prencentiles)

    # Find best radius
    best_radius = find_best_radius(train_val_dist, y_train, y_val, radii)

    print(f"Best radius selected: {best_radius:.4f}")

    predictions = nnr_predict_weighted(train_test_dist, y_train, best_radius)

    return predictions




students = {'id1': '211398003', 'id2': '322867524l'}

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
