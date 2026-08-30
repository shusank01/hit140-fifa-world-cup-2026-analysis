import pandas as pd
import numpy as np
from scipy import stats


# Load the dataset
try:
    df = pd.read_csv(
        "../data-set/fifa_player_data.txt",
        skiprows=1
    )
except FileNotFoundError:
    print("dataset file not found.")
    raise

print("Total records:", len(df))

df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
df["CrdY"] = pd.to_numeric(df["CrdY"], errors="coerce")

df = df.drop_duplicates()

played = df[df["Min"] > 0].copy()


# Defenders
defenders = played[played["Pos"] == "DF"].copy()
defenders = defenders.dropna(subset=["CrdY"])
defenders = defenders[defenders["CrdY"] >= 0]

print("\nEligible defenders:", len(defenders))


# Defender data check
print("\nMissing values of defenders:")
print(defenders[["Pos", "Min", "CrdY"]].isnull().sum())

print("Duplicate rows of defenders:", defenders.duplicated().sum())


# Defender sample
defenders_sample = defenders.sample(
    n=100,
    random_state=42
)

def_cards = defenders_sample["CrdY"]

print("Defender sample size:", len(def_cards))


# Defender statistics
print("\nDescriptive statistics (Defenders):")
print("Mean:", round(def_cards.mean(), 3))
print("Median:", round(def_cards.median(), 3))
print("Standard deviation:", round(def_cards.std(), 3))
print("Variance:", round(def_cards.var(), 3))
print("Minimum:", round(def_cards.min(), 3))
print("Maximum:", round(def_cards.max(), 3))


# 95% confidence interval
def_mean = def_cards.mean()
def_sd = def_cards.std()
def_n = len(def_cards)

def_se = def_sd / np.sqrt(def_n)
def_margin = 1.960 * def_se

def_lower = def_mean - def_margin
def_upper = def_mean + def_margin

print("\n95% confidence interval:")
print(round(def_lower, 3), "to", round(def_upper, 3))


# Midfielders
midfielders = played[played["Pos"] == "MF"].copy()
midfielders = midfielders.dropna(subset=["CrdY"])
midfielders = midfielders[midfielders["CrdY"] >= 0]

print("\nEligible midfielders:", len(midfielders))


# Midfielder data check
print("\nMissing values of midfielders:")
print(midfielders[["Pos", "Min", "CrdY"]].isnull().sum())

print("Duplicate rows of midfielders:", midfielders.duplicated().sum())


# Midfielder sample
midfielders_sample = midfielders.sample(
    n=100,
    random_state=42
)

mid_cards = midfielders_sample["CrdY"]

print("Midfielder sample size:", len(mid_cards))


# Midfielder statistics
print("\nDescriptive statistics (Midfielders):")
print("Mean:", round(mid_cards.mean(), 3))
print("Median:", round(mid_cards.median(), 3))
print("Standard deviation:", round(mid_cards.std(), 3))
print("Variance:", round(mid_cards.var(), 3))
print("Minimum:", round(mid_cards.min(), 3))
print("Maximum:", round(mid_cards.max(), 3))


# Comparing defenders and midfielders
t_stat, p_value = stats.ttest_ind(
    mid_cards,
    def_cards,
    equal_var=False
)

print("\nTwo-sample t-test (Midfielders vs Defenders):")
print("t-statistic:", round(t_stat, 4))
print("p-value:", round(p_value, 4))

alpha = 0.05

if p_value < alpha:
    print(
        "Reject H0: there is a statistically significant "
        "difference in yellow cards between midfielders and defenders."
    )
else:
    print(
        "Fail to reject H0: there is not enough evidence "
        "of a difference between midfielders and defenders."
    )


# Interpretation
print("\nInterpretation:")

if p_value < alpha:
    result = "indicates a statistically significant difference between the two groups."
else:
    result = "does not provide sufficient evidence of a difference between the two groups."

print(
    f"Defenders averaged {def_cards.mean():.3f} yellow cards "
    f"(95% CI: {def_lower:.3f} to {def_upper:.3f}), while midfielders "
    f"averaged {mid_cards.mean():.3f}. The two-sample t-test "
    f"(t = {t_stat:.3f}, p = {p_value:.4f}) {result}"
)