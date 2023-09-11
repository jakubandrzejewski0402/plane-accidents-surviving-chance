import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_data():
    le_region = LabelEncoder()
    le_plane_type = LabelEncoder()
    le_operator = LabelEncoder()
    le_time_of_day = LabelEncoder()

    data = pd.read_csv("aviation_accidents.csv")

    data["Month"] = pd.to_datetime(data["Date"]).dt.month
    data["Region"] = le_region.fit_transform(data["Location"].str.split(", ").str[-1])
    data["DayOfWeek"] = pd.to_datetime(data["Date"], format="%B %d, %Y").dt.dayofweek
    data["OperatorType"] = data["Operator"].apply(
        lambda x: "Military" if "military" in str(x).lower() else "Civilian"
    )
    data["OperatorType"] = le_operator.fit_transform(data["OperatorType"])
    data["PlaneType"] = data["Type"].str.split(" ").str[0]
    data["PlaneType"] = le_plane_type.fit_transform(data["PlaneType"])
    data["Time"] = pd.to_numeric(data["Time"], errors="coerce")
    data["Time"] = data["Time"].fillna(data["Time"].mean())
    data["TimeOfDay"] = data["Time"].apply(
        lambda x: "Day" if 8 <= int(x) / 100 < 20 else "Night"
    )
    data["TimeOfDay"] = le_time_of_day.fit_transform(data["TimeOfDay"])

    data["OperatorType"].fillna(data["OperatorType"].mode()[0], inplace=True)
    data["PlaneType"].fillna(data["PlaneType"].mode()[0], inplace=True)
    data["TimeOfDay"].fillna(data["TimeOfDay"].mode()[0], inplace=True)

    data["SurvivalRate"] = (data["Aboard"] - data["Fatalities"]) / data["Aboard"]
    data["SurvivalRate"].fillna(data["SurvivalRate"].mean(), inplace=True)

    data.dropna(subset=["Date", "Aboard", "Fatalities"], inplace=True)

    return data, le_region, le_plane_type, le_operator, le_time_of_day
