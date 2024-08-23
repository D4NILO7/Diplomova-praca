import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_wp_percentage_plot():
    files = ['evaluation_without_npc', 'evaluation_with_npc']
    file_names = ['evaluácia bez vozidiel', 'evaluácia s vozidlami']
    dataframes = []
    for file in files:
        df = pd.read_csv(file, usecols=['Model', 'avg_wp_reached'])
        # df['Model'] = df['Model'].apply(lambda x: x.split('\\')[-1].split('___')[0])
        dataframes.append(df)

    combined_df = pd.concat(dataframes)
    grouped_df = combined_df.groupby('Model', as_index=False).mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = grouped_df['Model'].unique()
    x = range(len(models))

    width = 0.35

    for i, file in enumerate(files):
        df = dataframes[i]
        df = df.groupby('Model', as_index=False).mean()
        df = df.set_index('Model').reindex(models).reset_index()
        bars = ax.bar([p + i * width for p in x], df['avg_wp_reached'], width=width, label=f'{file_names[i]}')

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{round(yval, 2)}%', ha='center', va='bottom')

    ax.set_xlabel('Model')
    ax.set_ylabel('Priemerné percento dosiahnutých navigačných bodov (%)')
    ax.set_title('Porovnávanie modelov z hľadiska percenta dokončenej trasy')
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(models)
    ax.legend()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_model_performance(data_path, column_name):
    df = pd.read_csv(data_path)

    df['Model'] = df['Model'].apply(lambda x: x.split('\\')[-1].split('___')[0])

    fig, ax = plt.subplots(figsize=(6, 6))

    bars = ax.bar(df['Model'], df[column_name], color='blue')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{round(yval, 2)}', ha='center', va='bottom')

    ax.set_xlabel('Model')
    ax.set_ylabel(column_name)
    ax.set_title(f'Performance of Models based on {column_name}')

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_normalized_data(data_path, columns, graph_width=10, graph_height=6):
    df = pd.read_csv(data_path)

    df['Model'] = df['Model'].apply(lambda x: x.split('\\')[-1].split('___')[0])

    for column in columns:
        df[f'normalized_{column}'] = df[column] / df[column].max()

    fig, ax = plt.subplots(figsize=(graph_width, graph_height))

    total_width = 0.8
    single_width = total_width / len(columns)
    bar_positions = [i - total_width / 2 + single_width / 2 for i in range(len(df))]

    for i, column in enumerate(columns):
        normalized_column = f'normalized_{column}'
        bars = ax.bar([x + i * single_width for x in bar_positions], df[normalized_column], width=single_width, label=column)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{round(df[column].iloc[bars.index(bar)], 2)}', ha='center', va='bottom')

    ax.set_xlabel('model')
    ax.set_ylabel('odmena')
    ax.set_title('Porovnanie modelov na základe získanej odmeny (s vozidlami)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Model'])

    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_normalized_data_three(data_path, columns, graph_width=10, graph_height=6):
    if len(columns) != 3:
        raise ValueError("This function requires exactly three columns for comparison.")

    df = pd.read_csv(data_path)

    df['Model'] = df['Model'].apply(lambda x: x.split('\\')[-1].split('___')[0])

    for column in columns:
        df[f'normalized_{column}'] = df[column] / df[column].max()

    fig, ax = plt.subplots(figsize=(graph_width, graph_height))

    total_width = 0.8
    single_width = total_width / len(columns)
    bar_positions = [i - total_width / 2 + single_width / 2 for i in range(len(df))]

    for i, column in enumerate(columns):
        normalized_column = f'normalized_{column}'
        bars = ax.bar([x + i * single_width for x in bar_positions], df[normalized_column], width=single_width,
                      label=column)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{round(df[column].iloc[bars.index(bar)], 2)}',
                    ha='center', va='bottom')

    ax.set_xlabel('Model')
    # ax.set_ylabel('Normalized Metrics')
    ax.set_title('Výkonnosť modelov z hľadiska jazdných úloh')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Model'])

    plt.xticks(rotation=0)
    ax.set_yticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    generate_wp_percentage_plot()
    plot_normalized_data("evaluation_without_npc", ["celková odmena", "priemerná odmena za akciu"], 10, 6)
    plot_normalized_data("evaluation_with_npc", ["celková odmena", "priemerná odmena za akciu"], 10, 6)
    plot_normalized_data_three("evaluation_without_npc", ["priemerná rýchlosť (km/h)","trvanie cesty (s)", "prekročenia čiar"], 10, 6)