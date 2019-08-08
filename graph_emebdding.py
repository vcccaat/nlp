import random
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas
import ujson as json
import node2vec
import networkx as nx
from gensim.models import Word2Vec
# from tqdm import tqdm


def get_neighbourhood_score(local_model, node1, node2):
    # cosine distance 
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return random.random()

def get_G_from_edges(edges):
    edge_dict = dict()
    # calculate the count for all the edges
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        # add edges to the graph
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        # add weights for all the edges
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G

def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)




# Start to load the train data

train_edges = list()
raw_train_data = pandas.read_csv('/Users/sze/Downloads/Project2/train.csv')
for i, record in raw_train_data.iterrows():
    train_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the train data.')

# Start to load the valid/test data

valid_positive_edges = list()
valid_negative_edges = list()
raw_valid_data = pandas.read_csv('/Users/sze/Downloads/Project2/valid.csv')
for i, record in raw_valid_data.iterrows():
    if record['label']:
        valid_positive_edges.append((str(record['head']), str(record['tail'])))
    else:
        valid_negative_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the valid data.')


directed = True
p = 0.5
q = 2
num_walks = 5  #each node walk number 5-20
walk_length = 20  #how long the walk path to be 
dimension = 10 #300 for text 
window_size = 10
num_workers = 4
iterations = 10

# Create a node2vec object with training edges
G = node2vec.Graph(get_G_from_edges(train_edges), directed, p, q)   #p=0.5 q=2 in paper
# Calculate the probability for the random walk process
G.preprocess_transition_probs()
# Conduct the random walk process
walks = G.simulate_walks(num_walks, walk_length)
# Train the node embeddings with gensim word2vec package
model = Word2Vec(walks, size=dimension, window=window_size, min_count=0, sg=1, workers=num_workers, iter=iterations) #small network should at least 10
# Save the resulted embeddings (you can use any format you like)
resulted_embeddings = dict()
for i, w in enumerate(model.wv.index2word):
    resulted_embeddings[w] = model.wv.syn0[i]


tmp_AUC_score = get_AUC(model, valid_positive_edges, valid_negative_edges)
print('tmp_accuracy:', tmp_AUC_score)
# 0.939794340944211

print('start to predict test data.')
test_edges = list()
raw_test_data = pandas.read_csv('/Users/sze/Downloads/Project2/test.csv')

test_prediction_list = list()
for i, record in raw_test_data.iterrows():
    test_edges.append((str(record['head']), str(record['tail'])))
for edge in test_edges:
    test_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
    test_prediction_list.append(test_score)

# Remember to change the path of saving file 
scores = np.array(test_prediction_list)
raw_test_data['score'] = scores
raw_test_data.to_csv('test0.csv',index=False)


print('end')
