import pandas as pd
import matplotlib.pyplot as plt
import logging
import os.path
import seaborn as sns

DATAFILE = './corrected_preprocessed_urls.csv'
ID_COLUMN = 'url'
LABEL = 'status'
FEATURES = {
    "url_length": int,
    "num_digits": int,
    "digit_ratio": float,
    "special_char_ratio": float,
    "num_hyphens": int,
    "num_underscores": int,
    "num_slashes": int,
    "num_dots": int,
    "num_question_marks": int,
    "num_equals": int,
    "num_at_symbols": int,
    "num_percent": int,
    "num_hashes": int,
    "num_ampersands": int,
    "num_subdomains": int,
    "is_https": bool,
    "has_suspicious_word": bool,
}

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.debug("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))
    return basename

def reduce_dataframe(df, rows=10000):
    rdf = df.copy()
    rdf = rdf.dropna()
    rdf = rdf.drop_duplicates()
    rdf = rdf.sample(n=rows)
    return rdf

def display_label_vs_feature(df, feature_column, figure_number):
    """
    Display a plot of label vs feature for a single feature.
    """
    fig = plt.figure(figure_number, figsize=(6.5, 9))
    fig.suptitle( "Label vs. Feature" )
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(feature_column, LABEL, data=df, s=1)
    ax.set_xlabel(feature_column)
    ax.set_ylabel(LABEL)
    ax.locator_params(axis='both', tight=True, nbins=5)
    fig.tight_layout()
    basename = get_basename(DATAFILE)
    figure_name = "{}-{}-scatter.{}".format(basename, feature_column, "pdf")
    fig.savefig('./plots/' + figure_name)
    plt.close(fig)
    return

def display_feature_histogram(df, feature_column, figure_number):
    """
    Display a histogram for a single feature.
    """
    fig = plt.figure(figure_number, figsize=(6.5, 9))
    fig.suptitle( "Feature Histogram" )
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale("log")
    n, bins, patches = ax.hist(df[feature_column], bins=20)
    ax.set_xlabel(feature_column)
    ax.locator_params(axis='x', tight=True, nbins=5)
    fig.tight_layout()
    basename = get_basename(DATAFILE)
    figure_name = "{}-{}-histogram.{}".format(basename, feature_column, "pdf")
    fig.savefig('./plots/' + figure_name)
    plt.close(fig)
    return

def display_feature_kdeplot(df, feature_column, figure_number):
    """
    Display a kdeplot for a single feature.
    """
    fig = plt.figure(figure_number, figsize=(6.5, 9))
    fig.suptitle( "Feature KDE Plot" )
    ax = fig.add_subplot(1, 1, 1)
    sns.kdeplot(x=df[feature_column], y=df[LABEL])
    ax.set_xlabel(feature_column)
    ax.locator_params(axis='x', tight=True, nbins=5)
    fig.tight_layout()
    basename = get_basename(DATAFILE)
    figure_name = "{}-{}-kdeplot.{}".format(basename, feature_column, "pdf")
    fig.savefig('./plots/' + figure_name)
    plt.close(fig)
    return

def shuffle_data(df, seed=42):
    return df.sample(frac=1, random_state=seed)

def split_data(df, ratio=0.2, seed=42):
    train = df.sample(frac=1-ratio, random_state=seed)
    test = df.drop(train.index)
    return train, test

def read_data():
    df = pd.read_csv(DATAFILE)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def main():
    df = read_data()
    train, test = split_data(shuffle_data(df))
    train.to_csv('./train.csv', index=False)
    test.to_csv('./test.csv', index=False)
    
    rdf = reduce_dataframe(train)
    f = list(FEATURES.keys())
    
    rdf = rdf.drop(columns=[ID_COLUMN])
    
    # rdf = rdf.drop(columns=['has_suspicious_word'])
    # f.remove('has_suspicious_word')
    
    # convert has_suspicious_word to int
    rdf['has_suspicious_word'] = rdf['has_suspicious_word'].astype(int)
    # convert is_https to int
    rdf['is_https'] = rdf['is_https'].astype(int)
    
    
    for feature in f:
        # display_feature_histogram(rdf, feature, 1)
        # display_label_vs_feature(rdf, feature, 2)
        display_feature_kdeplot(rdf, feature, 3)

if __name__ == "__main__":
    main()
