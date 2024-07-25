import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from schema import PostGet
from datetime import datetime
import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("model")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


model = load_models()


def load_features() -> pd.DataFrame:
    query = '''SELECT * FROM posts_info'''
    df_users_features = batch_load_sql(query)
    return df_users_features



def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://user:password@host:name"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql_query(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

df_fu = load_features() 
conn_uri = "postgresql://user:password@host:name"
post_text_df = pd.read_sql("SELECT * FROM post_text_df", conn_uri)
user_data = pd.read_sql("SELECT * FROM user_data", conn_uri)


df = df_fu.copy()
app = FastAPI()


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    post = []
    user_id = id
    element = user_data.loc[user_data['user_id']==id]
    element = element.drop(columns=['user_id'])
    n = len(post_text_df)
    element = element.loc[element.index.repeat(n)].reset_index(drop=True)
    data = pd.to_datetime(time)
    data = pd.DataFrame({'time_of_day': [data]*n})
    data['day_of_week'] = data['time_of_day'].dt.day_name()
    data['time_of_day'] = pd.cut(data['time_of_day'].dt.hour, bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'day', 'evening'])
    result = pd.concat([element, data], axis=1)
    result = pd.concat([result, df], axis=1)
    post_id = result[['post_id']]
    topic = result[['topic']]
    text = result[['text']]
    result = result[['time_of_day', 'day_of_week', 'topic', 'TotalTfIdf', 'MaxTfIdf', 'MeanTfIdf', 'TextCluster','DistanceTo1thCluster','DistanceTo2thCluster', 'DistanceTo3thCluster', 'DistanceTo4thCluster', 'DistanceTo5thCluster', 'like/viev', 'gender', 'age', 'country', 'city','exp_group', 'os', 'source']]
    answer = pd.DataFrame({'chance': model.predict_proba(result)[:, 1], 'text' : text['text'], 'id': post_id['post_id'], 'topic': topic['topic']})
    answer.sort_values('chance', inplace=True)
    for index, row in answer[['text', 'id', 'topic']].tail(limit).iterrows():
        data = PostGet( id=row['id'] , text=row['text'], topic=row['topic'])
        post.append(data)
    return post