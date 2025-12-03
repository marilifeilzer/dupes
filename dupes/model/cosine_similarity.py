from sklearn.metrics.pairwise import cosine_similarity

def extracting_feature_X(df):
    X= df[['encoded_ingriedients', 'encoded_categories']]




S = cosine_similarity()
