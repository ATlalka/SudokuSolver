import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def print_plots_by_method(filename, plot_dir):
    df = pd.read_csv(filename, names=['Algorithm', 'ID', 'DR', 'Time'])

    df['DR'] = df['DR'].astype(float)

    grouped = df.groupby(['Algorithm', 'DR'])['Time'].mean().reset_index()

    for method in grouped['Algorithm'].unique():
        method_data = grouped[grouped['Algorithm'] == method]
        plt.figure(figsize=(35, 6))
        plt.plot(method_data['DR'], method_data['Time'], marker='o')
        plt.title(f'Method: {method}')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Average time [s]')
        plt.grid(True)
        plt.xticks(method_data['DR'])
        plt.margins(x=0.01)

        output_filename = os.path.join(plot_dir, f'plot_{method}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()


def print_by_grouped_dr(filename, plot_dir):
    df = pd.read_csv(filename, names=['Algorithm', 'ID', 'DR', 'Time'])

    df['DR'] = df['DR'].astype(float)

    df['DR'] = df['DR'].round().astype(int)

    plt.figure(figsize=(15, 6))

    for method in df['Algorithm'].unique():
        method_data = df[df['Algorithm'] == method]

        grouped = method_data.groupby('DR')['Time'].mean()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Average time depending on method and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Average time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'mean_plot_grouped_methods.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in df['Algorithm'].unique():
        method_data = df[df['Algorithm'] == method]

        grouped = method_data.groupby('DR')['Time'].max()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Maximum time depending on method and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Maximum time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'max_plot_grouped_methods.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in df['Algorithm'].unique():
        method_data = df[df['Algorithm'] == method]

        grouped = method_data.groupby('DR')['Time'].min()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Minimum time depending on method and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Minimum time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'min_plot_grouped_methods.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()


# Function for creating a plot for each model
def print_plots_by_method_model(filename, plot_dir):
    df = pd.read_csv(filename, names=['Model', 'ID', 'DR', 'Time', 'Correctness'])

    df['DR'] = df['DR'].astype(float)
    df['Correctness'] = df['Correctness'].astype(float).round(2)

    grouped = df.groupby(['Model', 'DR'])['Time'].mean().reset_index()
    grouped_correctness = df.groupby(['Model', 'DR'])['Correctness'].mean().reset_index()

    for method in grouped['Model'].unique():
        method_data = grouped[grouped['Model'] == method]
        plt.figure(figsize=(35, 6))
        plt.plot(method_data['DR'], method_data['Time'], marker='o')
        plt.title(f'Model: {method}')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Average time [s]')
        plt.grid(True)
        plt.xticks(method_data['DR'])
        plt.margins(x=0.01)

        output_filename = os.path.join(plot_dir, f'plot_{method}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()

    for method in grouped_correctness['Model'].unique():
        method_data = grouped_correctness[grouped_correctness['Model'] == method]
        plt.figure(figsize=(35, 6))
        plt.plot(method_data['DR'], method_data['Correctness'], marker='o')
        plt.title(f'Model: {method}')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Correctness [%]')
        plt.grid(True)
        plt.xticks(method_data['DR'])
        plt.margins(x=0.01)

        output_filename = os.path.join(plot_dir, f'plot_{method}_correctness.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()


# Function for creating plots (mean, max, min) in which are all models
def print_by_grouped_dr_model(filename, plot_dir):
    df = pd.read_csv(filename, names=['Model', 'ID', 'DR', 'Time', 'Correctness'])

    df['DR'] = df['DR'].astype(float)

    df['DR'] = df['DR'].round().astype(int)

    plt.figure(figsize=(15, 6))

    for method in df['Model'].unique():
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Time'].mean()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Average time depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Average time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'mean_plot_grouped_models.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in df['Model'].unique():
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Time'].max()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Maximum time depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Maximum time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'max_plot_grouped_models.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in df['Model'].unique():
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Time'].min()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Minimum time depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Minimum time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'min_plot_grouped_models.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in df['Model'].unique():
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Correctness'].mean()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Average correctness depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Average correctness [%]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'mean_plot_grouped_models_correctness.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in df['Model'].unique():
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Correctness'].max()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Maximum correctness depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Maximum correctness [%]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'max_plot_grouped_methods_models_correctness.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in df['Model'].unique():
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Correctness'].min()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Minimum correctness depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Minimum correctness [%]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'min_plot_grouped_methods_models_correctness.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()


# Function for creating plots (mean, max, min) for each type of NN
def print_by_grouped_dr_model_and_type(filename, plot_dir):
    df = pd.read_csv(filename, names=['Model', 'ID', 'DR', 'Time', 'Correctness'])

    df['DR'] = df['DR'].astype(float)
    df['DR'] = df['DR'].round().astype(int)

    df['Correctness'] = df['Correctness'].astype(float).round(2)

    types = ['lstm', 'cnn', 'srnn']

    for type in types:
        filtered_df = df[df['Model'].str.contains(type, case=False)]

        plt.figure(figsize=(15, 6))

        for method in filtered_df['Model'].unique():
            method_data = df[df['Model'] == method]

            grouped = method_data.groupby('DR')['Time'].mean()

            plt.plot(grouped.index, grouped.values, marker='o', label=method)

        plt.title('Average time depending on model and difficulty rate')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Average time [s]')
        plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
        plt.grid(True)
        plt.margins(x=0.01)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists(f'{plot_dir}/{type}'):
            os.makedirs(f'{plot_dir}/{type}')

        output_filename = os.path.join(f'{plot_dir}/{type}', f'mean_plot_grouped_models_{type}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 6))

        for method in filtered_df['Model'].unique():
            method_data = df[df['Model'] == method]

            grouped = method_data.groupby('DR')['Time'].max()

            plt.plot(grouped.index, grouped.values, marker='o', label=method)

        plt.title('Maximum time depending on model and difficulty rate')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Maximum time [s]')
        plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
        plt.grid(True)
        plt.margins(x=0.01)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists(f'{plot_dir}/{type}'):
            os.makedirs(f'{plot_dir}/{type}')

        output_filename = os.path.join(f'{plot_dir}/{type}', f'max_plot_grouped_models_{type}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 6))

        for method in filtered_df['Model'].unique():
            method_data = df[df['Model'] == method]

            grouped = method_data.groupby('DR')['Time'].min()

            plt.plot(grouped.index, grouped.values, marker='o', label=method)

        plt.title('Minimum time depending on model and difficulty rate')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Minimum time [s]')
        plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
        plt.grid(True)
        plt.margins(x=0.01)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists(f'{plot_dir}/{type}'):
            os.makedirs(f'{plot_dir}/{type}')

        output_filename = os.path.join(f'{plot_dir}/{type}', f'min_plot_grouped_models_{type}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 6))

        for method in filtered_df['Model'].unique():
            method_data = df[df['Model'] == method]

            grouped = method_data.groupby('DR')['Correctness'].mean()

            plt.plot(grouped.index, grouped.values, marker='o', label=method)

        plt.title('Average correctness depending on model and difficulty rate')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Average correctness [%]')
        plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
        plt.grid(True)
        plt.margins(x=0.01)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists(f'{plot_dir}/{type}'):
            os.makedirs(f'{plot_dir}/{type}')

        output_filename = os.path.join(f'{plot_dir}/{type}', f'mean_plot_grouped_models_correctness_{type}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 6))

        for method in filtered_df['Model'].unique():
            method_data = df[df['Model'] == method]

            grouped = method_data.groupby('DR')['Correctness'].max()

            plt.plot(grouped.index, grouped.values, marker='o', label=method)

        plt.title('Maximum correctness depending on model and difficulty rate')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Maximum correctness [%]')
        plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
        plt.grid(True)
        plt.margins(x=0.01)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists(f'{plot_dir}/{type}'):
            os.makedirs(f'{plot_dir}/{type}')

        output_filename = os.path.join(f'{plot_dir}/{type}',
                                       f'max_plot_grouped_methods_models_correctness_{type}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 6))

        for method in filtered_df['Model'].unique():
            method_data = df[df['Model'] == method]

            grouped = method_data.groupby('DR')['Correctness'].min()

            plt.plot(grouped.index, grouped.values, marker='o', label=method)

        plt.title('Minimum correctness depending on model and difficulty rate')
        plt.xlabel('Difficulty rate')
        plt.ylabel('Minimum correctness [%]')
        plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
        plt.grid(True)
        plt.margins(x=0.01)
        plt.legend()
        plt.tight_layout()

        if not os.path.exists(f'{plot_dir}/{type}'):
            os.makedirs(f'{plot_dir}/{type}')

        output_filename = os.path.join(f'{plot_dir}/{type}',
                                       f'min_plot_grouped_methods_models_correctness_{type}.png')
        plt.savefig(output_filename,
                    bbox_inches='tight')
        plt.close()


def print_by_grouped_dr_model_extra(filename, filename2, plot_dir, model_names):
    df1 = pd.read_csv(filename, names=['Model', 'ID', 'DR', 'Time', 'Correctness'])
    df2 = pd.read_csv(filename2, names=['Model', 'ID', 'DR', 'Time', 'Correctness'])

    df = pd.concat([df1, df2], ignore_index=True)

    df['DR'] = df['DR'].astype(float)

    df['DR'] = df['DR'].round().astype(int)

    plt.figure(figsize=(15, 6))

    for method in model_names:
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Time'].mean()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Average time depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Average time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'bests_mean_plot_grouped_models.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in model_names:
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Time'].max()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Maximum time depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Maximum time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'bests_max_plot_grouped_models.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in model_names:
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Time'].min()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Minimum time depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Minimum time [s]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'bests_min_plot_grouped_models.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in model_names:
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Correctness'].mean()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Average correctness depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Average correctness [%]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'bests_mean_plot_grouped_models_correctness.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in model_names:
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Correctness'].max()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Maximum correctness depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Maximum correctness [%]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'bests_max_plot_grouped_methods_models_correctness.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 6))

    for method in model_names:
        method_data = df[df['Model'] == method]

        grouped = method_data.groupby('DR')['Correctness'].min()

        plt.plot(grouped.index, grouped.values, marker='o', label=method)

    plt.title('Minimum correctness depending on model and difficulty rate')
    plt.xlabel('Difficulty rate')
    plt.ylabel('Minimum correctness [%]')
    plt.xticks(range(df['DR'].min(), df['DR'].max() + 1))
    plt.grid(True)
    plt.margins(x=0.01)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(plot_dir, f'bests_min_plot_grouped_methods_models_correctness.png')
    plt.savefig(output_filename,
                bbox_inches='tight')
    plt.close()
