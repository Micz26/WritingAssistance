import pandas as pd

from writing_assistance.utils.normalize_data import normalize_avg_score
from writing_assistance.utils.evaluate_model import (
    generate_predictions,
    evaluate_formality_predictions,
    plot_scatter_true_vs_predicted,
    plot_error_histogram,
)


def main():
    detector_name = input("Enter the name of the model ('deberta', 'xlm_roberta', 'gpt'): ").lower()
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    df = pd.read_csv('hf://datasets/osyvokon/pavlick-formality-scores/' + splits['train'])

    normalize_avg_score(df)
    df_limited = df.sample(frac=1, random_state=42).head(100)

    y_true, y_pred = generate_predictions(df_limited, detector_name)

    print(evaluate_formality_predictions(y_true, y_pred))
    plot_scatter_true_vs_predicted(y_true, y_pred)
    plot_error_histogram(y_true, y_pred)


if __name__ == '__main__':
    main()
