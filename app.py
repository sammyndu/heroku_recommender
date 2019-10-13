import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sqlalchemy as sql
from flask import Flask, jsonify

app = Flask(__name__)

def post(p_id):
    """ Function to get an article title from the title field,
    given a post ID 
    Args: The post id
    Arg type: Integer
    Returns: A particular post title with id = post_id"""
    return POSTS[POSTS['post_id'] == p_id]['title'].tolist()[0].split(' - ')[0]

def recommend(post_id, num, RESULTS, POSTS):
    """ Function to reads the results out of the dictionary.
    Args: post id and number of recommendations required
    Arg type: Integer, Integer
    Returns: Dictionary of similar titles and score"""
    
    try:
        recs = RESULTS[post_id][:num]
    except KeyError:
        return {"msg": "post with id "+str(post_id)+" doesnt exist"}
    dic = []
    for rec in recs:
        print(dic.append(POSTS.loc[POSTS.post_id == rec[1]].to_dict('records')[0]))
    
    return dic
# Just plug in any post id here (we have about 800 posts in the dataset), and the number of recommendations you want (1-99)
# You can get a list of valid post IDs by evaluating the variable 'POSTS'


@app.route("/<int:post_id>", methods=['GET'])
def index(post_id):
    
    #ENGINE = sql.create_engine('mysql+mysqlconnector://root:@localhost/lucid')

    #checking out the post table
    #POSTS = pd.read_sql_query('select * from posts', ENGINE)
    POSTS = pd.read_json('https://github.com/hngi/team_python/blob/master/json_data/lucid_table_posts.json')
    POSTS.drop(['user_id', 'tags', 'slug', 'created_at', 'updated_at', 'image',
                'status_id', 'action', 'post_id'], axis=1, inplace=True)
    POSTS.rename(columns={"id":"post_id"}, inplace=True)
    POSTS.head(50)

    TF = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    TFIDF_MATRIX = TF.fit_transform(POSTS['title'])

    COSINE_SIMILARITIES = linear_kernel(TFIDF_MATRIX, TFIDF_MATRIX)

    RESULTS = {}

    for idx, row in POSTS.iterrows():
        similar_indices = COSINE_SIMILARITIES[idx].argsort()[:-100:-1]
        similar_items = [(COSINE_SIMILARITIES[idx][i], POSTS['post_id'][i]) for i in similar_indices]

        # First post is the post itself, so remove it.
        # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, post_id)
        RESULTS[row['post_id']] = similar_items[1:]
    result = recommend(post_id, 5, RESULTS, POSTS)
    if  isinstance(result, dict):
        response = jsonify(result)
        response.status_code = 404
        return response
    else:
        response = jsonify(result)
        response.status_code = 200
        return response
    

if __name__ == "__main__":
    app.run(debug=True)
