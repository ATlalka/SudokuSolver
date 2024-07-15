import os

from research import results_visualize, model_tests, regular_tests

if __name__ == '__main__':
    models_dir = 'models/v2_e10'
    models_mix_dir = 'models/mix_e10'

    models = ['lstm1', 'lstm2', 'lstm3', 'lstm4', 'srnn1', 'srnn2', 'srnn3', 'srnn4', 'cnn1',
              'cnn2', 'cnn3', 'cnn4']
    models_mix = ['cl64', 'lc64', 'cl128', 'lc128']
    best_models = ['cnn2', 'lstm2', 'lstm4', 'cl64']

    regular_methods_filename = 'results/results-v3.csv'
    model_results = 'results/model-e10-results.csv'
    model_mix_results = 'results/model-e10-mix-results.csv'

    plot_regular_methods_dir = 'plots'
    plot_model_dir = 'plots/models/e10'
    plot_model_mix_dir = 'plots/models/mix/e10'

    plot_best_models_dir = 'plots/models/best'

    if not os.path.exists(plot_model_dir):
        os.makedirs(plot_model_dir)

    if not os.path.exists(plot_model_mix_dir):
        os.makedirs(plot_model_mix_dir)

    if not os.path.exists(plot_regular_methods_dir):
        os.makedirs(plot_regular_methods_dir)

    if not os.path.exists(plot_best_models_dir):
        os.makedirs(plot_best_models_dir)

    regular_tests.test_single_position(regular_methods_filename)
    regular_tests.test_single_candidate(regular_methods_filename)
    regular_tests.test_backtracking(regular_methods_filename)

    model_tests.test_models(model_results, models, models_dir)
    model_tests.test_models(model_mix_results, models_mix, models_mix_dir)

    results_visualize.print_plots_by_method(regular_methods_filename, plot_regular_methods_dir)
    results_visualize.print_by_grouped_dr(regular_methods_filename, plot_regular_methods_dir)

    results_visualize.print_by_grouped_dr_model(model_results, plot_model_dir)
    results_visualize.print_plots_by_method_model(model_results, plot_model_dir)
    results_visualize.print_by_grouped_dr_model_and_type(model_results, plot_model_dir)

    results_visualize.print_by_grouped_dr_model(model_mix_results, plot_model_mix_dir)
    results_visualize.print_plots_by_method_model(model_mix_results, plot_model_mix_dir)

    results_visualize.print_by_grouped_dr_model_extra(model_results, model_mix_results, plot_best_models_dir,
                                                      best_models)
