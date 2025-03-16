import numpy
import numpy as np
import pandas as pd
import pickle
import argparse


# Pour tout information dans les packages AzureML: https://docs.microsoft.com/en-us/python/api/?view=azure-ml-py
from azureml.training.tabular._diagnostics import logging_utilities


def setup_instrumentation(automl_run_id):
    import logging
    import sys

    from azureml.core import Run
    from azureml.telemetry import INSTRUMENTATION_KEY, get_telemetry_log_handler
    from azureml.telemetry._telemetry_formatter import ExceptionFormatter

    logger = logging.getLogger("azureml.training.tabular")

    try:
        logger.setLevel(logging.INFO)

        # Ajouter les logs à la sortie du modéle
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        
        telemetry_handler = get_telemetry_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY, component_name="azureml.training.tabular"
        )
        telemetry_handler.setFormatter(ExceptionFormatter())
        logger.addHandler(telemetry_handler)
        try:
            run = Run.get_context() 
            # Pour faire l'affichage des metrics Les métriques enregistrées via run.log 
            # ou run.log_list sont automatiquement visibles sur l'interface utilisateur Azure ML dans l'onglet "Metrics" du Run correspondant.
            return logging.LoggerAdapter(logger, extra={
                "properties": {
                    "codegen_run_id": run.id,
                    "automl_run_id": automl_run_id
                }
            })
        except Exception:
            pass
    except Exception:
        pass

    return logger
"""
	Les métriques enregistrées apparaissent sous forme de tableaux ou de graphiques dans la section dédiée du Run.
    Azure ML génère automatiquement des graphiques pour les métriques enregistrées, avec des options pour explorer
      l'évolution des scores au fil des epochs ou sur différentes combinaisons de paramètres.
    En résumé, Azure ML facilite le suivi et la visualisation des métriques via son API RUN et son interface utilisateur.
    Si vous avez besoin de graphiques spécifiques, vous pouvez les créer localement avec Matplotlib, puis les intégrer dans le Run.
"""

automl_run_id = 'dmroversampling_20' # Dans la suite, je l'utilise conmme l'identifiant de mon expérience
logger = setup_instrumentation(automl_run_id)


def split_dataset(X, y, weights, split_ratio, should_stratify):
    '''
    Divise l'ensemble de données en un ensemble d'entraînement et de test.

    Divise l'ensemble de données à l'aide du ratio de division donné. Le ratio par défaut donné est de 0,25, mais peut être
    modifié dans la fonction principale. Si should_stratify est vrai, les données seront divisées de manière stratifiée, 
    ce qui signifie que chaque nouvel ensemble aura la même distribution de la valeur cible que l'ensemble de données d'origine.
    should_stratify est vrai pour une exécution de classification, sinon faux.
    '''
    from sklearn.model_selection import train_test_split

    random_state = 42
    if should_stratify:
        stratify = y
    else:
        stratify = None

    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
        weights_train, weights_test = None, None

    return (X_train, y_train, weights_train), (X_test, y_test, weights_test)


def get_training_dataset(dataset_uri):
    
    from azureml.core.run import Run
    from azureml.data.abstract_dataset import AbstractDataset
    
    logger.info("Running get_training_dataset")
    ws = Run.get_context().experiment.workspace
    dataset = AbstractDataset._load(dataset_uri, ws)
    return dataset.to_pandas_dataframe()


def prepare_data(dataframe):
    '''
    Prépare les données pour la formation.

    Nettoie les données, sépare les colonnes de pondération des caractéristiques et des échantillons et prépare les données pour une utilisation dans la formation.
    Cette fonction peut varier en fonction du type de jeu de données et du type de tâche d'expérimentation : classification,
    régression ou prévision de séries chronologiques.
    '''
    
    from azureml.training.tabular.preprocessing import data_cleaning
    
    logger.info("Running prepare_data")
    label_column_name = 'label' # Le nom de mon label
    
    # J'extrait ma colonne cible 'label' et ma colonne de 'commentaire_usages' puis je supprime les lignes vides
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    sample_weights = None
    X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
     is_timeseries=False, target_column=label_column_name)
    
    return X, y, sample_weights


def get_mapper_0(column_names):
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from numpy import float32
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': TfidfVectorizer,
                'analyzer': 'char',
                'binary': False,
                'decode_error': 'strict',
                'dtype': numpy.float32,
                'encoding': 'utf-8',
                'input': 'content',
                'lowercase': True,
                'max_df': 0.95,
                'max_features': None,
                'min_df': 1,
                'ngram_range': (3, 3),
                'norm': 'l2',
                'preprocessor': None,
                'smooth_idf': True,
                'stop_words': None,
                'strip_accents': None,
                'sublinear_tf': False,
                'token_pattern': '(?u)\\b\\w\\w+\\b',
                'tokenizer': None,
                'use_idf': False,
                'vocabulary': None,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def get_mapper_1(column_names):
    from azureml.training.tabular.featurization.text.stringcast_transformer import StringCastTransformer
    from numpy import float32
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn_pandas.dataframe_mapper import DataFrameMapper
    from sklearn_pandas.features_generator import gen_features
    
    definition = gen_features(
        columns=column_names,
        classes=[
            {
                'class': StringCastTransformer,
            },
            {
                'class': TfidfVectorizer,
                'analyzer': 'word',
                'binary': False,
                'decode_error': 'strict',
                'dtype': numpy.float32,
                'encoding': 'utf-8',
                'input': 'content',
                'lowercase': True,
                'max_df': 1.0,
                'max_features': None,
                'min_df': 1,
                'ngram_range': (1, 2),
                'norm': 'l2',
                'preprocessor': None,
                'smooth_idf': True,
                'stop_words': None,
                'strip_accents': None,
                'sublinear_tf': False,
                'token_pattern': '(?u)\\b\\w\\w+\\b',
                'tokenizer': None,
                'use_idf': False,
                'vocabulary': None,
            },
        ]
    )
    mapper = DataFrameMapper(features=definition, input_df=True, sparse=True)
    
    return mapper
    
    
def generate_data_transformation_config():
    '''
        Spécifie l'étape de caractérisation dans le pipeline scikit-learn final.

        Si vous avez de nombreuses colonnes qui doivent avoir la même caractérisation/transformation appliquée (par exemple,
        50 colonnes dans plusieurs groupes de colonnes), ces colonnes sont gérées par regroupement en fonction du type. Chaque
        groupe de colonnes possède alors un mappeur unique appliqué à toutes les colonnes du groupe.
    '''
    from sklearn.pipeline import FeatureUnion
    
    column_group_1 = ['commentaire_usager']
    
    feature_union = FeatureUnion([
        ('mapper_0', get_mapper_0(column_group_1)),
        ('mapper_1', get_mapper_1(column_group_1)),
    ])
    return feature_union
    
    
def generate_preprocessor_config_0():
    '''
    Spécifie une étape de prétraitement à effectuer après la caractérisation dans le pipeline scikit-learn final.

    Normalement, cette étape de prétraitement consiste uniquement en une standardisation/normalisation des données qui est
    réalisée avec sklearn.preprocessing. Le ML automatisé spécifie uniquement une étape de prétraitement pour
    les modèles de classification et de régression non-ensemble.
    '''
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=False,
        with_std=False
    )
    
    return preproc
    
    
def generate_algorithm_config_0():
    from xgboost.sklearn import XGBClassifier
    
    algorithm = XGBClassifier(
        base_score=0.5,
        booster='gbtree',
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=0.5,
        enable_categorical=False,
        eta=0.5,
        gamma=0,
        gpu_id=-1,
        importance_type=None,
        interaction_constraints='',
        learning_rate=0.5,
        max_delta_step=0,
        max_depth=6,
        max_leaves=3,
        min_child_weight=1,
        missing=numpy.nan,
        monotone_constraints='()',
        n_estimators=10,
        n_jobs=0,
        num_parallel_tree=1,
        objective='multi:softprob',
        predictor='auto',
        random_state=0,
        reg_alpha=0.7291666666666667,
        reg_lambda=2.3958333333333335,
        scale_pos_weight=None,
        subsample=0.8,
        tree_method='auto',
        use_label_encoder=True,
        validate_parameters=1,
        verbose=-10,
        verbosity=0
    )
    
    return algorithm
    
    
def generate_preprocessor_config_1():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=False,
        with_std=False
    )
    
    return preproc
    
    
def generate_algorithm_config_1():
    from xgboost.sklearn import XGBClassifier
    
    algorithm = XGBClassifier(
        base_score=0.5,
        booster='gbtree',
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=0.6,
        enable_categorical=False,
        eta=0.3,
        gamma=0,
        gpu_id=-1,
        importance_type=None,
        interaction_constraints='',
        learning_rate=0.300000012,
        max_delta_step=0,
        max_depth=6,
        max_leaves=0,
        min_child_weight=1,
        missing=numpy.nan,
        monotone_constraints='()',
        n_estimators=10,
        n_jobs=0,
        num_parallel_tree=1,
        objective='multi:softprob',
        predictor='auto',
        random_state=0,
        reg_alpha=0.3125,
        reg_lambda=2.3958333333333335,
        scale_pos_weight=None,
        subsample=1,
        tree_method='auto',
        use_label_encoder=True,
        validate_parameters=1,
        verbose=-10,
        verbosity=0
    )
    
    return algorithm
    
    
def generate_preprocessor_config_2():
    from sklearn.preprocessing import Normalizer
    
    preproc = Normalizer(
        copy=True,
        norm='l2'
    )
    
    return preproc
    
    
def generate_algorithm_config_2():
    from sklearn.ensemble import RandomForestClassifier
    
    algorithm = RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight='balanced',
        criterion='gini',
        max_depth=None,
        max_features='sqrt',
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=0.01,
        min_samples_split=0.01,
        min_weight_fraction_leaf=0.0,
        monotonic_cst=None,
        n_estimators=100,
        n_jobs=-1,
        oob_score=True,
        random_state=None,
        verbose=0,
        warm_start=False
    )
    
    return algorithm
    
    
def generate_algorithm_config():
    '''
    Spécifie l'algorithme réel et les hyperparamètres pour la formation du modèle.
    Il s'agit de la dernière étape du pipeline final de scikit-learn. Pour les modèles d'ensemble, generate_preprocessor_config_N()
    (si nécessaire) et generate_algorithm_config_N() sont définis pour chaque apprenant dans le modèle d'ensemble,
    où N représente le placement de chaque apprenant dans la liste du modèle d'ensemble. Pour l'ensemble de pile
    '''
    from azureml.training.tabular.models.voting_ensemble import PreFittedSoftVotingClassifier
    from numpy import array
    from sklearn.pipeline import Pipeline
    
    pipeline_0 = Pipeline(steps=[('preproc', generate_preprocessor_config_0()), ('model', generate_algorithm_config_0())])
    pipeline_1 = Pipeline(steps=[('preproc', generate_preprocessor_config_1()), ('model', generate_algorithm_config_1())])
    pipeline_2 = Pipeline(steps=[('preproc', generate_preprocessor_config_2()), ('model', generate_algorithm_config_2())])
    algorithm = PreFittedSoftVotingClassifier(
        classification_labels=numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), # L'encodage de ma colonne cible
        estimators=[
            ('model_0', pipeline_0),
            ('model_1', pipeline_1),
            ('model_2', pipeline_2),
        ],
        flatten_transform=False,
        weights=[0.4, 0.2, 0.4]
    )
    
    return algorithm
    
    
def build_model_pipeline():
    '''
    Defines the scikit-learn pipeline steps.
    '''
    from sklearn.pipeline import Pipeline
    
    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('featurization', generate_data_transformation_config()),
            ('ensemble', generate_algorithm_config()),
        ]
    )
    
    return pipeline


def train_model(X, y, sample_weights=None, transformer=None):
    '''
    Calls the fit() method to train the model.
    
    The return value is the model fitted/trained on the input data.
    '''
    
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    
    model = model_pipeline.fit(X, y)
    return model


def calculate_metrics(model, X, y, sample_weights, X_test, y_test, cv_splits=None):
    '''
    Calcule des mesures qui peuvent être utilisées pour évaluer les performances du modèle.
    '''
    
    from azureml.training.tabular.score.scoring import score_classification
    
    y_pred_probs = model.predict_proba(X_test)
    if isinstance(y_pred_probs, pd.DataFrame):
        y_pred_probs = y_pred_probs.values
    class_labels = np.unique(y)
    train_labels = model.classes_
    metrics = score_classification(
        y_test, y_pred_probs, get_metrics_names(), class_labels, train_labels, use_binary=True)
    return metrics

# Je liste tous les métrics pour évaluer mon modéle
def get_metrics_names():
    
    metrics_names = [
        'accuracy',
        'f1_score_macro',
        'iou_classwise',
        'classification_report',
        'matthews_correlation',
        'iou_micro',
        'balanced_accuracy',
        'accuracy_table',
        'iou_weighted',
        'recall_score_micro',
        'AUC_classwise',
        'precision_score_weighted',
        'average_precision_score_classwise',
        'precision_score_binary',
        'confusion_matrix',
        'AUC_binary',
        'iou_macro',
        'recall_score_binary',
        'f1_score_micro',
        'precision_score_classwise',
        'precision_score_micro',
        'average_precision_score_weighted',
        'f1_score_classwise',
        'average_precision_score_micro',
        'recall_score_classwise',
        'recall_score_macro',
        'log_loss',
        'norm_macro_recall',
        'precision_score_macro',
        'f1_score_weighted',
        'f1_score_binary',
        'iou',
        'AUC_weighted',
        'weighted_accuracy',
        'recall_score_weighted',
        'average_precision_score_binary',
        'average_precision_score_macro',
        'AUC_macro',
        'AUC_micro',
    ]
    return metrics_names


def get_metrics_log_methods():
    
    metrics_log_methods = {
        'accuracy': 'log',
        'f1_score_macro': 'log',
        'iou_classwise': 'Skip',
        'classification_report': 'Skip',
        'matthews_correlation': 'log',
        'iou_micro': 'Skip',
        'balanced_accuracy': 'log',
        'accuracy_table': 'log_accuracy_table',
        'iou_weighted': 'Skip',
        'recall_score_micro': 'log',
        'AUC_classwise': 'Skip',
        'precision_score_weighted': 'log',
        'average_precision_score_classwise': 'Skip',
        'precision_score_binary': 'log',
        'confusion_matrix': 'log_confusion_matrix',
        'AUC_binary': 'log',
        'iou_macro': 'Skip',
        'recall_score_binary': 'log',
        'f1_score_micro': 'log',
        'precision_score_classwise': 'Skip',
        'precision_score_micro': 'log',
        'average_precision_score_weighted': 'log',
        'f1_score_classwise': 'Skip',
        'average_precision_score_micro': 'log',
        'recall_score_classwise': 'Skip',
        'recall_score_macro': 'log',
        'log_loss': 'log',
        'norm_macro_recall': 'log',
        'precision_score_macro': 'log',
        'f1_score_weighted': 'log',
        'f1_score_binary': 'log',
        'iou': 'Skip',
        'AUC_weighted': 'log',
        'weighted_accuracy': 'log',
        'recall_score_weighted': 'log',
        'average_precision_score_binary': 'log',
        'average_precision_score_macro': 'log',
        'AUC_macro': 'log',
        'AUC_micro': 'log',
    }
    return metrics_log_methods


def main(training_dataset_uri=None):
    '''
    Runs all functions defined above.
    '''
    
    from azureml.automl.core.inference import inference
    from azureml.core.run import Run
    
    import mlflow
    
    # Le code suivant est destiné à l’exécution de ce code dans le cadre d’une exécution de script AzureML.
    run = Run.get_context()
    
    df = get_training_dataset(training_dataset_uri)
    X, y, sample_weights = prepare_data(df)
    split_ratio = 0.25
    try:
        (X_train, y_train, sample_weights_train), (X_valid, y_valid, sample_weights_valid) = split_dataset(X, y, sample_weights, split_ratio, should_stratify=True)
    except Exception:
        (X_train, y_train, sample_weights_train), (X_valid, y_valid, sample_weights_valid) = split_dataset(X, y, sample_weights, split_ratio, should_stratify=False)
    model = train_model(X_train, y_train, sample_weights_train)
    
    metrics = calculate_metrics(model, X, y, sample_weights, X_test=X_valid, y_test=y_valid)
    metrics_log_methods = get_metrics_log_methods()
    print(metrics)
    for metric in metrics:
        if metrics_log_methods[metric] == 'None':
            logger.warning("Unsupported non-scalar metric {}. Will not log.".format(metric))
        elif metrics_log_methods[metric] == 'Skip':
            pass # Les métriques de prévision non scalaires et les métriques de classification non prises en charge ne sont pas enregistrées
        else:
            getattr(run, metrics_log_methods[metric])(metric, metrics[metric])
    cd = inference.get_conda_deps_as_dict(True)
    
    #Sauvegarder le modéle dans outputs/.
    signature = mlflow.models.signature.infer_signature(X, y) # gére l'entré et la sortie du modéle 
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='outputs/',
        conda_env=cd,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    
    run.upload_folder('outputs/', 'outputs/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset_uri', type=str, default='azureml://locations/f../DataOversampling/versions/1',     help='Default training dataset uri is populated from the parent run')
    args = parser.parse_args()
    
    try:
        main(args.training_dataset_uri)
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise



# Quand vous déployez des modèles MLFlow sur un point de terminaison en ligne, 
# vous n’avez pas besoin de fournir un script et un environnement de scoring, car les deux sont générés automatiquement. 