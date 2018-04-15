from datetime import date

quarters = {1: "Q4", 2: "Q4", 3: "Q4",
            4: "Q1", 5: "Q1", 6: "Q1",
            7: "Q2", 8: "Q2", 9: "Q2",
            10: "Q3", 11: "Q3", 12: "Q3"}


def get_quarter(date: date) -> str:
    month = date.month
    return quarters[month] + " " + str(date.year if quarters[month] != "Q4" else date.year - 1)