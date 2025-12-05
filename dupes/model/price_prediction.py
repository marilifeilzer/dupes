import pandas as pd
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import AdaBoostRegressor, VotingRegressor, GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, mutual_info_regression, VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def preprocess_data(df: pd.DataFrame):
    cols_to_keep = ['price_eur', 'volume_ml', 'propiedad', 'ingredients_formula']

    return df[cols_to_keep]

#     encoder_ordinal = OrdinalEncoder(
#         categories=feat_ordinal_values_sorted,
#         dtype= np.int64,
#         handle_unknown="use_encoded_value",
#         unknown_value=-1 # Considers unknown values as worse than "missing"
#     )

#     preproc_ordinal = make_pipeline(
#         SimpleImputer(strategy="constant", fill_value="missing"),
#         encoder_ordinal,
#         MinMaxScaler()
#     )

#     preproc_numerical = make_pipeline(
#         KNNImputer(),
#         MinMaxScaler()
#     )

#     preproc_transformer = make_column_transformer(
#         (preproc_numerical, make_column_selector(dtype_include=["int64", "float64"])),
#         (preproc_ordinal, feat_ordinal),
#         (preproc_nominal, feat_nominal),
#         remainder="drop"
#     )

#     preproc = make_pipeline(
#         preproc_transformer,
#         preproc_selector
#     )

#     return preproc



# # Create eval test just for early-stopping purposes (XGBOOST and Deep Learning)
# X_train, X_eval, y_train_log, y_eval_log = train_test_split(X, y_log, random_state=42)

# # Instantiate model
# model_xgb = XGBRegressor(max_depth=10,
#                          n_estimators=300,
#                          eval_metric=["rmse"],
#                          learning_rate=0.1)

# # Option 2: Use XGBoost Library to fit it
# # It allows you to use an `early_stopping` criteria with a Train/Val slit
# X_train_preproc = preproc.fit_transform(X_train, y_train_log)
# X_eval_preproc = preproc.transform(X_eval)

# model_xgb_early_stopping.fit(
#     X_train_preproc,
#     y_train_log,
#     verbose=False,
#     eval_set=[(X_train_preproc, y_train_log), (X_eval_preproc, y_eval_log)],
# )

# # Retrieve performance metrics
# results = model_xgb_early_stopping.evals_result()
# epochs = len(results['validation_0']["rmse"])
# x_axis = range(0, epochs)

# # Plot RMSLE loss
# fig, ax = plt.subplots()

# ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
# ax.plot(x_axis, results['validation_1']['rmse'], label='Val')
# ax.legend(); plt.ylabel('RMSE (of log)'); plt.title('XGBoost Log Loss')

# print("Best Validation Score", min(results['validation_1']['rmse']))
