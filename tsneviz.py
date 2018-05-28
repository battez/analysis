'''
tSNE visualising Word2Vec words - utility functions.

Credit to github user aneesha, author of this snippet:
https://gist.github.com/aneesha/da9216fb8d84245f7af6edaa14f4efa9#file-display_closestwords_tsnescatterplot-ipynb

Sub-region plotting - 
Credit to github user MiguelSteph, author of this snippet: 
http://migsena.com/build-and-visualize-word2vec-model-on-amazon-reviews/

'''

# from gensim.models.doc2vec import Doc2Vec
import gensim
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def plot_region(x_bounds, y_bounds, points):
    '''
    plot a sub-region of words in a tSNE reduction, for a dataframe: points
    '''
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1]) 
    ]
     
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


def display_closestwords_tsnescatterplot(model, word, vec_size=200):
    '''
    tSNE visualising Word2Vec function.
    '''
    arr = np.empty((0, vec_size), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.title('Most similar word vectors in the generated embeddings for term: "' \
        + word + '"')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.xlabel("tSNE dimension 1")
    plt.ylabel("tSNE dimension 2")
    plt.show()

    # Make a DF of the words and their coords, & show a zoomed in region:

    # count = 26756
    # word_vectors_matrix = np.ndarray(shape=(count, 200), dtype='float64')
    # word_list = []
    # i = 0
    # for word in model.vocab:
    #     word_vectors_matrix[i] = model[word]
    #     word_list.append(word)
    #     i = i+1
    #     if i == count:
    #         break

    # word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)

    # points = pd.DataFrame(
    # [
    #     (word, coords[0], coords[1]) 
    #     for word, coords in [
    #         (word, word_vectors_matrix_2d[word_list.index(word)])
    #         for word in word_list
    #     ]
    # ],
    # columns=["word", "x", "y"]
    # )
    # print (points.head(10))
    # print (points.tail(10))
    # plot_region(x_bounds=(-160.0, -140.0), y_bounds=(-160.0, -140.0), points=points)


