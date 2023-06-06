"""
In this script we define functions for the recommender web
application
"""
import pandas as pd
import numpy as np
import random

def recommend_nmf(new_user_query, nmf_model, ranked=10):
    """
    Filters and recommends the top ranked movies for any given input query based on a trained NMF model. 
    Returns a list of top ranked movie titles.
    """
   #load the dataset
    movie = pd.read_csv('data/movie.csv')
    

    # 1. construct new_user-item dataframe given the query
    Q_matrix = nmf_model.components_
    Q = pd.DataFrame(Q_matrix, columns= movie['title'],index=nmf_model.get_feature_names_out())    
    
    #convert new_user_query into dataframe
    new_user_dataframe =  pd.DataFrame(new_user_query,
                                           columns=movie['title'],
                                           index=['new_user_query']
                                           )
        
    #filling the missing values with 0
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)
    
    # 2. scoring
    P_new_user_matrix = nmf_model.transform(new_user_dataframe_imputed)
    # calculate the score with the NMF model
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q_matrix)
    
    # 3. ranking
    
    # filter out movies already seen by the user
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=movie['title'],
                         index = ['new_user'])

    R_hat_new_user_filtered =  R_hat_new_user.drop(new_user_query.keys(), axis=1)
        
    # return the top-k highest rated movie ids or titles
    ranked =  R_hat_new_user_filtered.T.sort_values(by =['new_user'],ascending=False).index.to_list()

    recommended = ranked[:3]
    
    return recommended

def recommender_nbcf(new_user_query, nbcf_model, df_score_ranked=10):
    """
    Filters and recommends the top ranked movies for any given input query based on a trained NMF model. 
    Returns a list of top ranked movie titles.
    """

    #load the dataset
    movie = pd.read_csv('data/movie.csv')
    rating = pd.read_csv('data/rating_pred.csv', index_col=0)
    
# 1. construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(new_user_query,
                                       columns=movie['title'],
                                       index=['new_user']
                                       )
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)
    
# 2. scoring

    # calculates the distances to all other users in the data!
    similarity_scores, neighbor_ids = nbcf_model.kneighbors(
        new_user_dataframe_imputed,
        n_neighbors=5,
        return_distance=True
    )
        # calculate the cosine similarity score with the NBCF model
    neighbors_df = pd.DataFrame(
    data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )
    
# 3. ranking
    # filter out movies already seen by the user
    neighborhood = rating.iloc[neighbor_ids[0]]
    neighborhood_filtered = neighborhood.drop(new_user_query.keys(), axis=1)
    
    # return the top-k highest rated movie ids or titles
    df_score = neighborhood_filtered.sum()
    df_score_ranked = df_score.sort_values(ascending=False).index.to_list()
    
    recommendation = df_score_ranked[:3]
    return recommendation

#def random_recommender(k=2):
#    if k > len(MOVIES):
#        print("Hey, you exceed the length of the movies")
#        return []
#    else:
#        random.shuffle(MOVIES)
#        top_k = MOVIES[:k]
#        return top_k

#if __name__ == "__main__":
#    top2 = random_recommender()
#    print(top2)