import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'

from sklearn.decomposition import PCA
import pandas as pd
import gensim
import gensim.downloader


def load_model(model_name, folder = "models", limit = 200000):
    """Downloads word2vec model from web or loads model from file if already downloaded"""
    binary_path = f"{folder}/{model_name}.bin"    
    
    try:
        # load model binary from file if it has been downloaded already
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(binary_path, binary=True, limit = limit)
        print(f"loaded from binary: {binary_path}")
    except:
        # download model from web and save to binary for faster loading next time
        w2v_model = gensim.downloader.load(model_name)
        w2v_model.save_word2vec_format(binary_path, binary=True, write_header=True)
    
    return w2v_model


def embed_words(words, w2v_model):
    return np.array([w2v_model[w] for w in words])

def reduce_dimensions(embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    return embeddings_reduced

def plot_embeddings(embeddings_reduced, words, title = "Reduced Embedding Space"):
    """

    """
    # Assemble into a data frame
    df = pd.DataFrame({"word": words, "x": embeddings_reduced[:,0], "y": embeddings_reduced[:,1], "z": embeddings_reduced[:,2]})

    # plot each embedding as a point with a label
    fig = px.scatter_3d(
        df, x = "x", y = "y", z = "z",
        color = "z", 
        text = "word", 
        width = 700, height = 600, 
        opacity = 0.7,
        title = title
    )
    
    # add lines from origin to point to mmake it look like a vector
    for word, coord in zip(words, embeddings_reduced):
        fig.add_trace(go.Scatter3d(
            x=[0, coord[0]], y=[0, coord[1]], z=[0, coord[2]],
            mode='lines',
            line_width = 1,
            line_color = "SlateGrey",
            showlegend = False
        ))
    fig.update_layout(uniformtext_minsize=6, uniformtext_mode='hide')
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    return fig


def analogy(word1, word2, word3, model, n=5):
    """
    Returns analogy word using the given model.

    Parameters
    --------------
    word1 : (str)  word1 in the analogy relation
    word2 : (str)  word2 in the analogy relation
    word3 : (str)  word3 in the analogy relation
    model : word2vec embedding model
    n : (int) the number of most similar words to return. Default is 5
    
    Returns
    ---------------
        pd.dataframe
    """
    print(f"{word1.upper()} is to  {word2.upper()} is as {word3.upper()} is to : ____")
    sim_words = model.most_similar(positive=[word3, word2], negative=[word1], topn = n)
    return pd.DataFrame(sim_words, columns=["Analogy word", "Score"])