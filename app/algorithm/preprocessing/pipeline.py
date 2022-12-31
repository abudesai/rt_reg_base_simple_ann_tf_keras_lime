from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler, OneHotEncoder, PowerTransformer
import sys, os
import joblib
import pandas as pd 
import algorithm.preprocessing.preprocessors as preprocessors



PREPROCESSOR_FNAME = "preprocessor.save"



'''

PRE-POCESSING STEPS =====>

=========== initial pre-processing ========
- Filter out 'info' variables

=========== for categorical variables ========
- Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories 
# NOT DONE =>>> - Categorical variables: convert categories to ordinal scale by correlating to target
- One hot encode categorical variables

=========== for numerical variables ========
- Add binary column to represent 'missing' flag for missing values
- Impute missing values with mean of non-missing
- MinMax scale variables prior to yeo-johnson transformation
- Use Yeo-Johnson transformation to get (close to) gaussian dist. 
- Standard scale data after yeo-johnson

=========== for target variable ========
- Use Yeo-Johnson transformation to get (close to) gaussian dist. 
- Standard scale target data after yeo-johnson
===============================================
'''

  

def get_preprocess_pipeline(pp_params, model_cfg): 
    
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]
    pp_cat_params = model_cfg["pp_params"]["cat_params"]
    
    pipe_steps = []
    
    # ===== ADD TARGET FEATURE COLUMN IF NOT PRESENT (THIS CAN HAPPEN FOR TEST ATA)   =====
    # feature engine doesnt like it if a column is missing, even if it is not needed for a given transformation. 
    pipe_steps.append(
        (
            pp_step_names["TARGET_FEATURE_ADDER"],
            preprocessors.TargetFeatureAdder(
                label_field_name=pp_params['target_attr_name']
                ),
        )
    )    
    
    
    # ===== KEEP ONLY COLUMNS WE USE   =====
    pipe_steps.append(
        (
            pp_step_names["COLUMN_SELECTOR"],
            preprocessors.ColumnSelector(
                columns=pp_params['retained_vars']
                ),
        )
    )
    
    # ===============================================================
    # ===== CATEGORICAL VARIABLES =====
    
    pipe_steps.append(
        # ===== CAST CAT VARS TO STRING =====
        (
            pp_step_names["STRING_TYPE_CASTER"],
            preprocessors.StringTypeCaster(
                cat_vars=pp_params['cat_vars']
                ),
        )
    )
        
    
    # impute categorical na with string 'missing'           
    if len(pp_params['cat_na_impute_with_str_missing']):
        pipe_steps.append(
            (
                pp_step_names["CAT_IMPUTER_MISSING"],
                CategoricalImputer(
                    imputation_method="missing",
                    variables=pp_params["cat_na_impute_with_str_missing"],
                ),
            )
        )
        
    # impute categorical na with most frequent category
    if len(pp_params['cat_na_impute_with_freq']):
        pipe_steps.append(
            (
                pp_step_names["CAT_IMPUTER_FREQ"],
                CategoricalImputer(
                    imputation_method="frequent",
                    variables=pp_params["cat_na_impute_with_freq"],
                ),
            )
        )
        
    if len(pp_params['cat_vars']):
        # rare-label encoder
        pipe_steps.append(
            (
                pp_step_names["CAT_RARE_LABEL_ENCODER"],
                RareLabelEncoder(
                    tol=pp_cat_params["rare_perc_threshold"], 
                    n_categories=1, 
                    variables=pp_params["cat_vars"]
                ),
            )
        )
        
        # one-hot encoder cat vars
        pipe_steps.append(
            (
                pp_step_names["ONE_HOT_ENCODER"],
                preprocessors.OneHotEncoderMultipleCols(                    
                    ohe_columns=pp_params["cat_vars"],
                ),
            )
        )
        
        # == DROP UNWANTED FEATURES 
        pipe_steps.append(
            (
                pp_step_names["FEATURE_DROPPER"],
                preprocessors.ColumnSelector(
                    columns=pp_params["cat_vars"],
                    selector_type='drop')
            )
        )
        
    # ===============================================================
    # ===== NUMERICAL VARIABLES =====
    
    pipe_steps.append(
        # ===== CAST CAT VARS TO STRING =====
        (
            pp_step_names["FLOAT_TYPE_CASTER"],
            preprocessors.FloatTypeCaster(
                num_vars=pp_params['num_vars']
                ),
        )
    )
    
    if len(pp_params['num_na']):
        # add missing indicator to nas in numerical features 
        pipe_steps.append(
            (
                pp_step_names["NUM_MISSING_INDICATOR"],
                AddMissingIndicator(variables=pp_params["num_na"]),
            )
        )
        # impute numerical na with the mean
        pipe_steps.append(
            (
                pp_step_names["NUM_MISSING_MEAN_IMPUTER"],
                MeanMedianImputer(
                    imputation_method="mean",
                    variables=pp_params["num_na"],
                )
            )
        )
    
    
    # Transform numerical variables - minmax scale, yeo-johnson, standard
    if len(pp_params['num_vars']):        
        
        # MinMaxScale numeric attributes
        pipe_steps.append(
            (
                pp_step_names["MIN_MAX_SCALER"],
                SklearnTransformerWrapper(                    
                    MinMaxScaler(),
                    variables=pp_params["num_vars"],
                ),
            )
        )                     
        
        # Yeo-Johnson transformation
        pipe_steps.append(
            (
                pp_step_names["YEO_JOHN_TRANSFORMER"],
                preprocessors.CustomYeoJohnsonTransformer(
                    cols_list=pp_params["num_vars"]
                )
            )
        )
        
        # Min max bound numeric attributes
        pipe_steps.append(
            (
                pp_step_names["MINMAX_BOUNDER"],
                preprocessors.MinMaxBounder(
                    cols_list=pp_params["num_vars"]
                ),
            )
        )  
        
        # Standard Scale num vars
        pipe_steps.append(
            (
                pp_step_names["STANDARD_SCALER"], 
                SklearnTransformerWrapper(                    
                    StandardScaler(),
                    variables=pp_params["num_vars"] 
                ),    
            )
        )  
        
        # Clip values to +/- 4 std devs
        pipe_steps.append(
            (
                pp_step_names["VALUE_CLIPPER"], 
                preprocessors.ValueClipper(
                    fields_to_clip=pp_params["num_vars"],
                    min_val=-4.0,   # - 4 std dev
                    max_val=4.0,    # + 4 std dev    
                ),    
            )
        )    
        
    
    # ===============================================================
    # ===== TARGET VARIABLE =====    
    
    # MinMaxScale numeric attributes
    pipe_steps.append(
        (
            pp_step_names["MIN_MAX_SCALER_TARGET"],
            preprocessors.CustomMinMaxScaler(
                cols_list=[pp_params["target_attr_name"]]
            )
        )
    )        
    
    # Yeo-Johnson transformation for target
    pipe_steps.append(
        (
            pp_step_names["YEO_JOHN_TRANSFORMER_TARGET"],
            preprocessors.CustomYeoJohnsonTransformer(
                cols_list=[pp_params["target_attr_name"]]
            )
        )
    )
    
    pipe_steps.append(
        (
            pp_step_names["MINMAX_BOUNDER_TARGET"],
            preprocessors.MinMaxBounder(
                cols_list=[pp_params["target_attr_name"]]
            ),
        )
    )         
    
    # Standard Scale Target
    pipe_steps.append(
        (
            pp_step_names["STANDARD_SCALER_TARGET"], 
            preprocessors.CustomStandardScaler(
                cols_list=[pp_params["target_attr_name"]]
            )
        )
    )   
    
    # Clip values to +/- 4 std devs
    pipe_steps.append(
        (
            pp_step_names["TARGET_VALUE_CLIPPER"], 
            preprocessors.ValueClipper(
                fields_to_clip=[pp_params["target_attr_name"]],
                min_val=-4.0,   # - 4 std dev
                max_val=4.0,    # + 4 std dev    
            ),    
        )
    )       
    
    
    # ===============================================================
    # xy Splitter
    pipe_steps.append(
        (
            pp_step_names["XYSPLITTER"], 
            preprocessors.XYSplitter(
                target_col=pp_params["target_attr_name"],
                id_col=pp_params["id_field"],
                ),
        )
    )  
    # ===============================================================    
      
    pipeline = Pipeline( pipe_steps )
    
    return pipeline


def get_inverse_transform_on_preds(pipeline, model_cfg, preds):
    
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
    
    std_scaler_lbl = pp_step_names['STANDARD_SCALER_TARGET']
    std_scaler = pipeline[std_scaler_lbl]
    preds = std_scaler.inverse_transform(preds)    
    
    mmbounder_scaler_lbl = pp_step_names['MINMAX_BOUNDER_TARGET']
    mmbounder_scaler = pipeline[mmbounder_scaler_lbl]
    preds = mmbounder_scaler.inverse_transform(preds)    
    
    
    yj_scaler_lbl = pp_step_names['YEO_JOHN_TRANSFORMER_TARGET']
    yj_scaler = pipeline[yj_scaler_lbl]
    preds = yj_scaler.inverse_transform(preds)    
    
    
    minmax_scaler_lbl = pp_step_names['MIN_MAX_SCALER_TARGET']    
    minmax_scaler = pipeline[minmax_scaler_lbl]    
    preds = minmax_scaler.inverse_transform(preds)
    
       
    return preds
    
    

def save_preprocessor(preprocess_pipe, file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    try: 
        joblib.dump(preprocess_pipe, file_path_and_name)   
    except: 
        raise Exception(f'''
            Error saving the preprocessor. 
            Does the file path exist {file_path}?''')  
    return    
    

def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    if not os.path.exists(file_path_and_name):
        raise Exception(f'''Error: No trained preprocessor found. 
        Expected to find model files in path: {file_path_and_name}''')
        
    try: 
        preprocess_pipe = joblib.load(file_path_and_name)     
    except: 
        raise Exception(f'''
            Error loading the preprocessor. 
            Do you have the right trained preprocessor at {file_path_and_name}?''')
    
    return preprocess_pipe 
    