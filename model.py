import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from prepare_data import get_data


data, le_region, le_plane_type, le_operator, le_time_of_day = get_data()

features = data[
    ["Region", "Month", "DayOfWeek", "OperatorType", "PlaneType", "TimeOfDay"]
]
labels = data["SurvivalRate"]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# param_grid = {
#     "n_estimators": [100, 200, 300, 400, 500],
#     "max_depth": [None, 10, 20, 30, 40, 50],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["auto", "sqrt", "log2"],
# }
#
# rf = RandomForestRegressor(random_state=42)
#
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1,
#     scoring='neg_mean_squared_error',
#     verbose=2
# )

# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# print(f"Best parameters: {best_params}")

# model = RandomForestRegressor(**best_params)

model = RandomForestRegressor(
    max_depth=10,
    max_features="auto",
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=500,
    random_state=42,
)

model.fit(X_train, y_train)

scores = cross_val_score(
    model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
)

print("Cross-Validation MSE Scores: ", -scores)
print("Average Cross-Validation MSE Score: ", -scores.mean())

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


def predict_survival_rate(
    region, month, day_of_week, operator_type, plane_type, time_of_day
):
    region_code = le_region.transform([region])[0]
    type_code = le_plane_type.transform([plane_type])[0]
    operator_code = le_operator.transform([operator_type])[0]
    time_of_day_code = le_time_of_day.transform([time_of_day])[0]

    input_data = pd.DataFrame(
        data=[
            [
                region_code,
                month,
                day_of_week,
                operator_code,
                type_code,
                time_of_day_code,
            ]
        ],
        columns=[
            "Region",
            "Month",
            "DayOfWeek",
            "OperatorType",
            "PlaneType",
            "TimeOfDay",
        ],
    )
    survival_rate = model.predict(input_data)
    return survival_rate[0]


def feature_importances():
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], color="orange", align="center")
    plt.xticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.xlabel("Features")
    plt.ylabel("Relative Importance")
    plt.xticks(rotation=45)

    plt.savefig("feature-importances.png")
    plt.tight_layout()
    plt.close()


def plot_survival_by_region():
    regions = [
        "Poland",
        "New York",
        "Germany",
        "Russia",
        "France",
        "China",
        "Brazil",
        "England",
        "Iran",
        "Ukraine",
        "Nigeria",
        "Spain",
        "Egypt",
    ]
    survival_rates = []

    for region in regions:

        survival_rate = predict_survival_rate(region, 7, 1, "Civilian", "Boeing", "Day")
        survival_rates.append(survival_rate)

    plt.figure(figsize=(10, 6))
    plt.bar(regions, survival_rates, color="orange")
    plt.xlabel("Region")
    plt.ylabel("Predicted Survival Rate")
    plt.title("Predicted Survival Rates by Region")
    plt.xticks(rotation=45)

    plt.savefig("survival_by_region.png")
    plt.close()


def plot_survival_by_plane_type():
    plane_types = [
        "Boeing",
        "Airbus",
        "Cessna",
        "Embraer",
        "Bombardier",
        "Douglas",
        "Beechcraft",
        "Antonov",
    ]
    survival_rates = []

    for plane_type in plane_types:

        survival_rate = predict_survival_rate(
            "Russia", 7, 1, "Civilian", plane_type, "Day"
        )
        survival_rates.append(survival_rate)

    plt.figure(figsize=(10, 6))
    plt.bar(plane_types, survival_rates, color="orange")
    plt.xlabel("Plane Type")
    plt.ylabel("Predicted Survival Rate")
    plt.title("Predicted Survival Rates by Plane Type")
    plt.xticks(rotation=45)

    plt.savefig("survival_by_plane_type.png")
    plt.close()


def plot_survival_by_month():
    months = list(range(1, 13))
    survival_rates = []

    for month in months:
        survival_rate = predict_survival_rate(
            "Germany", month, 5, "Military", "Boeing", "Night"
        )
        survival_rates.append(survival_rate)

    plt.figure(figsize=(10, 6))
    plt.bar(months, survival_rates, color="orange")
    plt.xlabel("Month")
    plt.ylabel("Predicted Survival Rate")
    plt.title("Predicted Survival Rates by Month")
    plt.xticks(months)

    plt.savefig("survival_by_month.png")
    plt.close()


def plot_mse_vs_estimators(estimators_range=(1, 500), step=50):
    mse_values = []
    estimators = list(range(estimators_range[0], estimators_range[1] + 1, step))

    for n in estimators:
        model = RandomForestRegressor(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_values.append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(estimators, mse_values, marker="o", color="orange")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs. Number of Estimators")
    plt.savefig("mse_vs_estimators.png")
    plt.close()


def plot_error_histogram():
    predictions = model.predict(X_test)
    errors = y_test - predictions

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.5, color="orange")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Errors")

    plt.savefig("error_histogram.png")
    plt.close()


if __name__ == "__main__":
    print(predict_survival_rate("Poland", 7, 2, "Civilian", "Airbus", "Day"))
    print(predict_survival_rate("Russia", 1, 6, "Military", "Cessna", "Night"))
