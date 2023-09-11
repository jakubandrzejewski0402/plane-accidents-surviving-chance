import matplotlib.pyplot as plt
import pandas as pd
from prepare_data import get_data

data, le_region, le_plane_type, le_operator, le_time_of_day = get_data()


def plot_missing_data_percentage():
    data_from_csv = pd.read_csv("./aviation_accidents.csv")

    data_from_csv = data_from_csv[
        ["Date", "Location", "Operator", "Type", "Time", "Aboard", "Fatalities"]
    ]

    missing_data = data_from_csv.isnull().sum() / len(data) * 100
    missing_data = missing_data.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(missing_data.index, missing_data.values, color="r", alpha=0.7)
    plt.title("Percentage of Missing Cells for Each Column")
    plt.xlabel("Columns")
    plt.ylabel("Percentage of Missing Cells")
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.005,
            round(yval, 2),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("missing_data")
    plt.close()


def accidents_by_region():
    accidents_per_region = (
        data["Region"].value_counts().sort_values(ascending=False).head(40)
    )
    region_labels = le_region.inverse_transform(accidents_per_region.index)

    plt.figure(figsize=(10, 6))
    plt.bar(region_labels, accidents_per_region, color="b", alpha=0.7)
    plt.title("Number of Air Accidents by Region (Top 40)")
    plt.xlabel("Region")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("accidents_by_region")
    plt.close()


def accidents_by_month():
    accidents_per_month = data["Month"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    accidents_per_month.plot(kind="bar", color="b", alpha=0.7)
    plt.title("Number of Air Accidents by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Accidents")
    plt.xticks(
        ticks=range(0, 12),
        labels=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        rotation=0,
    )

    plt.savefig("accidents_by_month")
    plt.close()


def accidents_by_plane_type():
    data["PlaneType"] = data["Type"].str.split(" ").str[0]
    accidents_per_plane_type = (
        data["PlaneType"].value_counts().sort_values(ascending=False).head(25)
    )
    plane_producer_labels = accidents_per_plane_type.index

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        plane_producer_labels, accidents_per_plane_type, color="b", alpha=0.7
    )
    plt.title("Number of Air Accidents by Plane Type")
    plt.xlabel("Plane Producer")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            yval,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("accidents_by_plane_producer")
    plt.close()


def accidents_by_day_of_week():
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    accidents_per_day = data["DayOfWeek"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(days, accidents_per_day, color="b", alpha=0.7)
    plt.title("Number of Air Accidents by Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Number of Accidents")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            yval,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("accidents_by_day_of_week")
    plt.close()


def accidents_by_operator_type():
    operator_types = ["Military", "Civilian"]
    accidents_per_operator_type = data["OperatorType"].value_counts()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(operator_types, accidents_per_operator_type, color="b", alpha=0.7)
    plt.title("Number of Air Accidents by Operator Type")
    plt.xlabel("Operator Type")
    plt.ylabel("Number of Accidents")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            yval,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("accidents_by_operator_type")
    plt.close()


def accidents_by_time_of_day():
    times_of_day = ["Day", "Night"]
    accidents_per_time_of_day = data["TimeOfDay"].value_counts()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(times_of_day, accidents_per_time_of_day, color="b", alpha=0.7)
    plt.title("Number of Air Accidents by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Accidents")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            yval,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("accidents_by_time_of_day")
    plt.close()


def survival_rate_by_month():
    survival_rate_per_month = data.groupby("Month")["SurvivalRate"].mean()

    plt.figure(figsize=(10, 6))
    survival_rate_per_month.plot(kind="bar", color="g", alpha=0.7)
    plt.title("Average Survival Rate by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Survival Rate")
    plt.xticks(
        ticks=range(0, 12),
        labels=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        rotation=0,
    )

    plt.savefig("survival_rate_by_month")
    plt.close()


def survival_rate_by_region():
    data["Region"] = le_region.inverse_transform(data["Region"])
    top_regions = data["Region"].value_counts().index[:30]
    data_top_regions = data[data["Region"].isin(top_regions)]
    survival_rate_per_region = (
        data_top_regions.groupby("Region")["SurvivalRate"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    survival_rate_per_region.plot(kind="bar", color="g", alpha=0.7)
    plt.title("Average Survival Rate by Region")
    plt.xlabel("Region")
    plt.ylabel("Average Survival Rate")
    plt.xticks(rotation=45)

    plt.savefig("survival_rate_by_region.png")
    plt.close()


def survival_rate_by_operator_type():
    data["OperatorType"] = le_operator.inverse_transform(data["OperatorType"])
    operator_types = ["Civilian", "Military"]
    survival_rate_per_operator_type = (
        data.groupby("OperatorType")["SurvivalRate"].mean().loc[operator_types]
    )

    plt.figure(figsize=(10, 6))
    survival_rate_per_operator_type.plot(kind="bar", color="g", alpha=0.7)
    plt.title("Average Survival Rate by Operator Type")
    plt.xlabel("Operator Type")
    plt.ylabel("Average Survival Rate")

    plt.savefig("survival_rate_by_operator_type.png")
    plt.close()


def survival_rate_by_plane_type():
    data["PlaneType"] = le_plane_type.inverse_transform(data["PlaneType"])

    top_plane_types = data["PlaneType"].value_counts().index[:30]
    data_top_plane_types = data[data["PlaneType"].isin(top_plane_types)]

    survival_rate_per_plane_type = (
        data_top_plane_types.groupby("PlaneType")["SurvivalRate"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    survival_rate_per_plane_type.plot(kind="bar", color="g", alpha=0.7)
    plt.title("Average Survival Rate by Plane Type")
    plt.xlabel("Plane Type")
    plt.ylabel("Average Survival Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("survival_rate_by_plane_type.png")
    plt.close()


def survival_rate_by_time_of_day():
    data["TimeOfDay"] = le_time_of_day.inverse_transform(data["TimeOfDay"])

    time_of_day_labels = ["Day", "Night"]
    survival_rate_per_time_of_day = (
        data.groupby("TimeOfDay")["SurvivalRate"].mean().loc[time_of_day_labels]
    )

    plt.figure(figsize=(10, 6))
    survival_rate_per_time_of_day.plot(kind="bar", color="g", alpha=0.7)
    plt.title("Average Survival Rate by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Average Survival Rate")

    plt.tight_layout()
    plt.savefig("survival_rate_by_time_of_day.png")
    plt.close()


def survival_rate_by_day_of_week():
    day_of_week_labels = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    survival_rate_per_day_of_week = (
        data.groupby("DayOfWeek")["SurvivalRate"].mean().reindex(range(7))
    )

    plt.figure(figsize=(10, 6))
    survival_rate_per_day_of_week.plot(kind="bar", color="g", alpha=0.7)
    plt.title("Average Survival Rate by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Survival Rate")
    plt.xticks(range(7), day_of_week_labels)

    plt.tight_layout()
    plt.savefig("survival_rate_by_day_of_week.png")
    plt.close()
