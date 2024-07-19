import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
data = pd.read_csv('AmesHousing.csv')

# transformer
class SelectColumns( BaseEstimator, TransformerMixin ):
    # pass the function we want to apply to the column 'SalePrice’
    def __init__( self, columns ):
        self.columns = columns
    # don't need to do anything
    def fit( self, xs, ys, **params ):
        return self
    # actually perform the selection
    def transform( self, xs ):
        return xs[ self.columns ]

# steps and pipe
steps = [
          ('column_select', SelectColumns('Overall Qual')),
          ('linear_regression', LinearRegression( n_jobs = -1 ))
]

pipe = Pipeline(steps); 

# setup target + features w/ dumdum categorical -> numerical
xs_one = data.drop( columns = ['SalePrice'] )
xs_one = xs_one.fillna(0)
ys_one = data['SalePrice']

# get the categoricl
x_dumdum = pd.get_dummies(xs_one['Kitchen Qual'], prefix='Kitchen Qual', prefix_sep='_', dtype=int)
xs_one = pd.concat([xs_one, x_dumdum], axis =1)

x_dumdum2 = pd.get_dummies(xs_one['Paved Drive'], prefix='Paved Drive', prefix_sep='_', dtype=int)
xs_one = pd.concat([xs_one, x_dumdum2], axis =1)

x_dumdum5 = pd.get_dummies(xs_one['Garage Finish'], prefix='Garage Finish', prefix_sep='_', dtype=int)
xs_one = pd.concat([xs_one, x_dumdum5], axis =1)

x_dumdum8 = pd.get_dummies(xs_one['Exter Qual'], prefix='Exter Qual', prefix_sep='_', dtype=int)
xs_one = pd.concat([xs_one, x_dumdum8], axis =1)

x_dumdum10 = pd.get_dummies(xs_one['Bldg Type'], prefix='Bldg Type', prefix_sep='_', dtype=int)
xs_one = pd.concat([xs_one, x_dumdum10], axis =1)

x_dumdum11 = pd.get_dummies(xs_one['Lot Config'], prefix='Bldg Type', prefix_sep='_', dtype=int)
xs_one = pd.concat([xs_one, x_dumdum11], axis =1)

x_dumdum12 = pd.get_dummies(xs_one['Bsmt Qual'], prefix='Bldg Type', prefix_sep='_', dtype=int)
xs_one = pd.concat([xs_one, x_dumdum12], axis =1)

#    Notes: 
#        to get a variable from a function
#        __ to access it such as in column select to get edit columns __
#
grid = { 
        'column_select__columns': [
             [
                'Overall Qual', 'Year Remod/Add', '1st Flr SF', 'Total Bsmt SF', 'Gr Liv Area',
                'Full Bath', 'Garage Area','Lot Area', 'Pool Area', 'Fireplaces',
                'Kitchen Qual_Ex', 'Kitchen Qual_Fa', 'Kitchen Qual_Gd',
                'Kitchen Qual_Po', 'Kitchen Qual_TA', 
                'Paved Drive_N', 'Paved Drive_P', 'Paved Drive_Y', 
                'Exter Qual_Ex', 'Exter Qual_Fa', 'Exter Qual_Gd', 'Exter Qual_TA',
                'Bldg Type_1Fam', 'Bldg Type_2fmCon', 'Bldg Type_Duplex',
                'Bldg Type_Twnhs', 'Bldg Type_TwnhsE', 
                'Bldg Type_Corner', 'Bldg Type_CulDSac', 'Bldg Type_FR2',
                'Bldg Type_FR3', 'Bldg Type_Inside', 
                'Bldg Type_0', 'Bldg Type_Ex', 'Bldg Type_Fa', 'Bldg Type_Gd',
                'Bldg Type_Po', 'Bldg Type_TA'
            ], 
        ],
        'linear_regression': [
            LinearRegression( n_jobs = -1 ), # no transformation
            TransformedTargetRegressor(
                LinearRegression( n_jobs = -1 ),
                func = np.sqrt,
                inverse_func = np.square ),
            TransformedTargetRegressor(
                LinearRegression( n_jobs = -1 ),
                func = np.cbrt,
                inverse_func = lambda y: np.power( y, 3 ) ),
            TransformedTargetRegressor(
                LinearRegression( n_jobs = -1 ),
                func = np.log,
                inverse_func = np.exp),
        ]
}

search = GridSearchCV( pipe, grid, scoring = 'r2', n_jobs = -1 )
search.fit(xs_one, ys_one)

print("Linear regression: ")
print(search.best_score_)
print(search.best_params_)
print()

# decisiontree 
steps = [
          ('column_select', SelectColumns('Overall Qual')),
          ('regressor', DecisionTreeRegressor())
]

pipe = Pipeline(steps); 

#    Notes: 
#        to get a variable from a function
#        __ to access it such as in column select to get edit columns __
#
grid = { 
        'column_select__columns': [
             [
                'Overall Qual', 'Year Remod/Add', '1st Flr SF', 'Total Bsmt SF', 'Gr Liv Area',
                'Full Bath', 'Garage Area','Lot Area', 'Pool Area', 'Fireplaces',
                'Kitchen Qual_Ex', 'Kitchen Qual_Fa', 'Kitchen Qual_Gd',
                'Kitchen Qual_Po', 'Kitchen Qual_TA', 
                'Paved Drive_N', 'Paved Drive_P', 'Paved Drive_Y', 
                'Exter Qual_Ex', 'Exter Qual_Fa', 'Exter Qual_Gd', 'Exter Qual_TA',
                'Bldg Type_1Fam', 'Bldg Type_2fmCon', 'Bldg Type_Duplex',
                'Bldg Type_Twnhs', 'Bldg Type_TwnhsE', 
                'Bldg Type_Corner', 'Bldg Type_CulDSac', 'Bldg Type_FR2',
                'Bldg Type_FR3', 'Bldg Type_Inside', 
                'Bldg Type_0', 'Bldg Type_Ex', 'Bldg Type_Fa', 'Bldg Type_Gd',
                'Bldg Type_Po', 'Bldg Type_TA'
            ], 
        ],
        "regressor__max_depth": range(7, 10), # best was 8 
        "regressor__min_samples_split": range(14, 17), #1-40  #best was 15 also start point is 2 not 1
        "regressor__min_samples_leaf": range(1, 2), #1-20 # best was 2 
}

search = GridSearchCV( pipe, grid, cv = 10, n_jobs=-1, scoring = 'r2' )
search.fit(xs_one, ys_one)

print("Decision tree: ")
print(search.best_score_)
print(search.best_params_)
print()

# RandomForestRegressor 
steps = [
          ('column_select', SelectColumns('Overall Qual')),
          ('regressor', RandomForestRegressor())
]


pipe = Pipeline(steps); 

#    Notes: 
#        to get a variable from a function
#        __ to access it such as in column select to get edit columns __
#
grid = { 
        'column_select__columns': [
             [
                'Overall Qual', 'Year Remod/Add', '1st Flr SF', 'Total Bsmt SF', 'Gr Liv Area',
                'Full Bath', 'Garage Area','Lot Area', 'Pool Area', 'Fireplaces',
                'Kitchen Qual_Ex', 'Kitchen Qual_Fa', 'Kitchen Qual_Gd',
                'Kitchen Qual_Po', 'Kitchen Qual_TA', 
                'Paved Drive_N', 'Paved Drive_P', 'Paved Drive_Y', 
                'Exter Qual_Ex', 'Exter Qual_Fa', 'Exter Qual_Gd', 'Exter Qual_TA',
                'Bldg Type_1Fam', 'Bldg Type_2fmCon', 'Bldg Type_Duplex',
                'Bldg Type_Twnhs', 'Bldg Type_TwnhsE', 
                'Bldg Type_Corner', 'Bldg Type_CulDSac', 'Bldg Type_FR2',
                'Bldg Type_FR3', 'Bldg Type_Inside', 
                'Bldg Type_0', 'Bldg Type_Ex', 'Bldg Type_Fa', 'Bldg Type_Gd',
                'Bldg Type_Po', 'Bldg Type_TA'
            ], 
        ],
        "regressor__max_depth": range(7, 10), # best was 8 
        "regressor__min_samples_split": range(14, 17), #1-40  #best was 15 also start point is 2 not 1
        "regressor__min_samples_leaf": range(1, 2), #1-20 # best was 2 
        
        "regressor__n_estimators": [6, 7], # 6 was best out of 2 4 6
        "regressor__max_features": ["log2", None], # log 2 was best
    
    
}

search = GridSearchCV( pipe, grid, cv = 10,  n_jobs=-1 , scoring = 'r2')
search.fit(xs_one, ys_one)

print("Random forest: ")
print(search.best_score_)
print(search.best_params_)
print()

# GradientBoostingRegressor 
steps = [
          ('column_select', SelectColumns('Overall Qual')),
          ('regressor', GradientBoostingRegressor())
]

pipe = Pipeline(steps); 

#    Notes: 
#        to get a variable from a function
#        __ to access it such as in column select to get edit columns __
#
grid = { 
        'column_select__columns': [
             [
                'Overall Qual', 'Year Remod/Add', '1st Flr SF', 'Total Bsmt SF', 'Gr Liv Area',
                'Full Bath', 'Garage Area','Lot Area', 'Pool Area', 'Fireplaces',
                'Kitchen Qual_Ex', 'Kitchen Qual_Fa', 'Kitchen Qual_Gd',
                'Kitchen Qual_Po', 'Kitchen Qual_TA', 
                'Paved Drive_N', 'Paved Drive_P', 'Paved Drive_Y', 
                'Exter Qual_Ex', 'Exter Qual_Fa', 'Exter Qual_Gd', 'Exter Qual_TA',
                'Bldg Type_1Fam', 'Bldg Type_2fmCon', 'Bldg Type_Duplex',
                'Bldg Type_Twnhs', 'Bldg Type_TwnhsE', 
                'Bldg Type_Corner', 'Bldg Type_CulDSac', 'Bldg Type_FR2',
                'Bldg Type_FR3', 'Bldg Type_Inside', 
                'Bldg Type_0', 'Bldg Type_Ex', 'Bldg Type_Fa', 'Bldg Type_Gd',
                'Bldg Type_Po', 'Bldg Type_TA'
            ], 
        ],
        "regressor__max_depth": [7], # best was 8 
        "regressor__min_samples_split": range(11, 14), #1-40  #best was 15 also start point is 2 not 1
        "regressor__min_samples_leaf": range(1, 3), #1-20 # best was 2 
        
        "regressor__n_estimators": [50], # 6 was best out of 2 4 6, more better time up tho 10 and 15 not much dif
        "regressor__max_features": ["log2", None], # log 2 was best
        
        "regressor__learning_rate": [0.1, 0.2, 0.3], #.4 best went up to .5 
        "regressor__loss": ["squared_error"]
}

search = GridSearchCV( pipe, grid, cv = 10, n_jobs=-1, scoring = 'r2')
search.fit(xs_one, ys_one)

print("Gradient boosting: ")
print(search.best_score_)
print(search.best_params_)