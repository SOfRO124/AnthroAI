import pandas as pd
import matplotlib.pyplot as plt


def separate_sentences(sents, nlp):
    """Split text into sentences using a spacy model passed as argument."""
    doc = nlp(sents)
    res = []
    for sent in doc.sents:
        res.append(sent.text.lower())
    return res


def get_scores_no_duplicates(input_file):
    text_scores = pd.read_csv(input_file)
    text_scores = (
        text_scores
        .drop_duplicates(subset=["sentence"])
        .sort_values(by=["text_id"])
        .reset_index(drop=True)
    )
    return text_scores


def plot_scores(df_eng, df_ger, title, plots_dir="plots/"):
    ger_results = df_ger["anthroscore"]
    eng_results = df_eng["anthroscore"]
    tmp = pd.DataFrame({"Deutsch": ger_results, "Englisch": eng_results})

    print("boxplot")
    print("ger:", ger_results.describe())
    print("-------------------")
    print("eng:", eng_results.describe())

    fig, axs = plt.subplots()
    axs = tmp.boxplot(column=["Deutsch", "Englisch"])
    axs.set_title(f"{title}")
    plt.show()
    fig.savefig(plots_dir + f"{title}.png")


def grouped_stats(all_dfs, output_file="generated_data/Anthroscore_results/grouped_by_terms.csv"):
    tmp = []
    for k, elem in all_dfs.items():
        try:
            grouped = elem.groupby(["original_term"])
            df_mean = grouped["anthroscore"].mean()
            df_mean.rename(k, inplace=True)
            tmp.append(df_mean)
        except Exception:
            print("anthroscore not found in", k)
    res = pd.DataFrame(tmp)
    res.to_csv(output_file)
    return res
