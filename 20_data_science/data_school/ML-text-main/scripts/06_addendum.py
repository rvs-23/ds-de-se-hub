# # Addendum to "Advanced Machine Learning Techniques"

# ## Starter code

import pandas as pd
import numpy as np


def make_features(df):
    df['num_ingredients'] = df.ingredients.apply(len)
    df['ingredient_length'] = df.ingredients.apply(lambda x: np.mean([len(item) for item in x]))
    df['ingredients_str'] = df.ingredients.astype('str')
    return df


train = make_features(pd.read_json('../data/train.json'))
y = train['cuisine']


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


vect = CountVectorizer(token_pattern=r"'([a-z ]+)'")
nb = MultinomialNB()


# ## Part 6 rewritten using `ColumnTransformer`

# [make_column_transformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html)

from sklearn.compose import make_column_transformer


# vectorize 1 column, passthrough 2 columns, and drop the remaining columns
ct = make_column_transformer(
    (vect, 'ingredients_str'),
    ('passthrough', ['num_ingredients', 'ingredient_length']),
    remainder='drop')


# create the feature matrix from the DataFrame
X_dtm_manual = ct.fit_transform(train)
X_dtm_manual.shape


# ### Cross-validation

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# create a pipeline of the ColumnTransformer and Naive Bayes
pipe = make_pipeline(ct, nb)


# properly cross-validate the entire pipeline
cross_val_score(pipe, train, y, cv=5, scoring='accuracy').mean()


# ### Alternative way to specify `Pipeline` and `ColumnTransformer`

# [Pipeline documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and [ColumnTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# duplicate the pipeline structure without using make_pipeline or make_column_transformer
pipe = Pipeline([
    ('columntransformer', ColumnTransformer([
            ('countvectorizer', vect, 'ingredients_str'),
            ('passthrough', 'passthrough', ['num_ingredients', 'ingredient_length'])],
            remainder='drop')),
    ('multinomialnb', nb)
])


# ### Grid search of a nested `Pipeline`

# examine the pipeline steps
pipe.steps


# create a grid of parameters to search (and specify the pipeline step along with the parameter)
param_grid = {}
param_grid['columntransformer__countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-z ]+)'"]
param_grid['multinomialnb__alpha'] = [0.5, 1]
param_grid


from sklearn.model_selection import GridSearchCV


# pass the pipeline to GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(train, y);


# print the single best score and parameters that produced that score
print(grid.best_score_)
print(grid.best_params_)
