import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler


def fit_single_factor_model(predictor, target):
    """
    Fit a single-factor linear model: target ~ predictor.

    Parameters
    ----------
    predictor : np.ndarray or pd.Series
        Array of predictor returns (e.g., SPY).
    target : np.ndarray or pd.Series
        Array of target returns (responder stock).

    Returns
    -------
    model : LinearRegression
        Fitted linear model.
    predictions : np.ndarray
        Predicted values from the model.
    """
    model = LinearRegression(fit_intercept=False)
    # Reshape if needed to ensure correct shape for sklearn
    X = (
        predictor.values.reshape(-1, 1)
        if hasattr(predictor, "values")
        else predictor.reshape(-1, 1)
    )
    y = target.values if hasattr(target, "values") else target

    model.fit(X, y)
    predictions = model.predict(X)
    return model, predictions


class MultiFactorLinearModel:
    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.selected_features_ = None
        self.final_model_ = None
        self.fallback_feature_ = None
        self.is_fallback_ = False
        self.feature_names_ = None
        self.scaler_ = None

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns
            X_values = X.values
        else:
            self.feature_names_ = np.arange(X.shape[1])
            X_values = X
        y_values = y.values if hasattr(y, "values") else y

        # First, scale all features for LassoCV
        full_scaler = StandardScaler()
        X_scaled_full = full_scaler.fit_transform(X_values)

        lasso = LassoCV(
            cv=self.cv,
            fit_intercept=False,
            random_state=self.random_state,
            max_iter=10000,
        )
        lasso.fit(X_scaled_full, y_values)
        coefs = lasso.coef_
        selected_mask = coefs != 0
        selected_features = self.feature_names_[selected_mask]

        if selected_features.size == 0:
            # Fallback to single-factor model
            # Compute correlations
            if hasattr(X, "columns"):
                corrs = X.corrwith(y)
                best_feature = corrs.abs().idxmax()
                best_feature_mask = X.columns == best_feature
            else:
                corrs = [
                    np.corrcoef(X_values[:, i], y_values)[0, 1]
                    for i in range(X_values.shape[1])
                ]
                best_idx = np.argmax(np.abs(corrs))
                best_feature = self.feature_names_[best_idx]
                best_feature_mask = self.feature_names_ == best_feature

            self.is_fallback_ = True
            self.fallback_feature_ = best_feature
            final_indices = np.where(best_feature_mask)[0]

            # Now fit a new scaler on the selected feature only
            self.scaler_ = StandardScaler()
            X_best_scaled = self.scaler_.fit_transform(X_values[:, final_indices])

            self.final_model_ = LinearRegression(fit_intercept=False)
            self.final_model_.fit(X_best_scaled, y_values)
            self.selected_features_ = [best_feature]
        else:
            # Multi-factor model with selected features
            final_indices = np.where(selected_mask)[0]

            # Fit a new scaler on just the selected features
            self.scaler_ = StandardScaler()
            X_selected_scaled = self.scaler_.fit_transform(X_values[:, final_indices])

            self.final_model_ = LinearRegression(fit_intercept=False)
            self.final_model_.fit(X_selected_scaled, y_values)
            self.selected_features_ = list(selected_features)

        return self

    def predict(self, X):
        if hasattr(X, "columns"):
            if self.is_fallback_:
                X_values = X[[self.fallback_feature_]].values
            else:
                X_values = X[self.selected_features_].values
        else:
            # Map selected_features_ back to indices
            feature_indices = [
                np.where(self.feature_names_ == f)[0][0]
                for f in self.selected_features_
            ]
            X_values = X[:, feature_indices]

        # Scale using the scaler fitted on selected features only
        X_scaled = self.scaler_.transform(X_values)
        y_pred = self.final_model_.predict(X_scaled)
        return y_pred
