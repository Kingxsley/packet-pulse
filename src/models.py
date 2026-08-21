"""Model training for unsupervised and supervised anomaly detectors."""
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from . import config


def train_isolation_forest(X_train_scaled, contamination: float):
    model = IsolationForest(contamination=contamination, **config.ISO_FOREST_PARAMS)
    model.fit(X_train_scaled)
    return model


def build_autoencoder(input_dim: int):
    # Imported lazily so the rest of the pipeline works even in environments
    # where TensorFlow isn't installed (it's the heaviest dependency here).
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Dense(16, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(input_dim, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(X_train_normal_scaled):
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_autoencoder(X_train_normal_scaled.shape[1])
    early_stopping = EarlyStopping(
        monitor="loss", patience=config.AUTOENCODER_PARAMS["patience"], restore_best_weights=True
    )
    model.fit(
        X_train_normal_scaled, X_train_normal_scaled,
        epochs=config.AUTOENCODER_PARAMS["epochs"],
        batch_size=config.AUTOENCODER_PARAMS["batch_size"],
        callbacks=[early_stopping], verbose=0,
    )
    return model


def autoencoder_scores(model, X_scaled):
    reconstruction = model.predict(X_scaled, verbose=0)
    return np.mean((X_scaled - reconstruction) ** 2, axis=1)


def train_random_forest(X_train, y_train, tune: bool = False):
    if not tune:
        return RandomForestClassifier(**config.RF_PARAMS).fit(X_train, y_train)

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight="balanced"),
        config.RF_TUNE_GRID, n_iter=config.TUNE_N_ITER, cv=config.TUNE_CV,
        scoring="f1", random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best Random Forest params: {search.best_params_}")
    return search.best_estimator_


def train_xgboost(X_train, y_train, tune: bool = False):
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    if not tune:
        params = dict(config.XGB_PARAMS, scale_pos_weight=scale_pos_weight)
        return XGBClassifier(**params).fit(X_train, y_train)

    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=config.RANDOM_STATE,
                       scale_pos_weight=scale_pos_weight),
        config.XGB_TUNE_GRID, n_iter=config.TUNE_N_ITER, cv=config.TUNE_CV,
        scoring="f1", random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best XGBoost params: {search.best_params_}")
    return search.best_estimator_
