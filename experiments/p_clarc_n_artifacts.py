import pandas as pd
import matplotlib.pyplot as plt

import rootutils
import wandb
import seaborn as sns

sns.set_style(style="white")

n_colors = 10  # Number of colors in the palette
color_palette = []
for pal in ["Blues", "Reds", "Oranges", "Greens", "Purples"]:
    palette = sns.color_palette(pal, n_colors)
    color_palette.append(palette.as_hex()[7])

sns.set_style(style="white")
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("dilyabareeva/r-clarc")

summary_list, config_list, name_list, commit_list = [], [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    commit_list.append(run.commit)

data = pd.DataFrame(
    {
        "summary": summary_list,
        "config": config_list,
        "commit": commit_list,
    }
)


# Function to flatten the dataframe
def flatten_df(df):
    flat_df = pd.DataFrame()
    for col in df.columns:
        flat_df = pd.concat([flat_df, pd.json_normalize(df[col])], axis=1)
    return flat_df


flat_df = flatten_df(data)
flat_df["name"] = name_list


condition = {
    "model.model_name": "vgg16",
    "data.dataset_name": "funnybirds_mult_artifacts_v2",
}

print(condition)

filtered_df = flat_df.copy()
for column, value in condition.items():
    filtered_df = filtered_df[filtered_df[column] == value]
filtered_df = filtered_df[filtered_df.apply(lambda row: "NART" in row["name"], axis=1)]
filtered_df["Number of Artifacts Suppressed"] = filtered_df["data.artifacts"].str.len()
filtered_df["Clean Samples Test Accuracy"] = filtered_df["test_accuracy_clean"]

# filtered_df.set_index(["model.model_name", "method.method"], inplace=True)
# filtered_df = filtered_df.applymap(lambda x: str.format("{:0_.3f}", x))


df = filtered_df.loc[filtered_df["method.method"] == "p-clarc"]
df["method"] = "P-ClArC"
df2 = filtered_df.loc[filtered_df["method.method"] == "r-clarc"]
df2["method"] = "Class-cond. R-ClArC"
df3 = filtered_df.loc[filtered_df["method.method"] == "acr-clarc"]
df3["method"] = "Artifact-cond. R-ClArC"
df4 = filtered_df.loc[filtered_df["method.method"] == "accr-clarc"]
df4["method"] = r"Class-Artifact-cond. R-ClArC"

df = pd.concat([df, df2, df3])
plt.figure(figsize=(3.29, 2.2))
fig = sns.lineplot(
    df,
    x="Number of Artifacts Suppressed",
    y="Clean Samples Test Accuracy",
    hue="method",
    palette=sns.color_palette(color_palette),
    markers=True,
    style="method",
    linewidth=1.0,
    markersize=6,
)

plt.tight_layout()
plt.legend(loc="best", markerscale=0.8)

plt.savefig("./results/acc_artifacts_figma.png", bbox_inches="tight", dpi=1000)
plt.show()
