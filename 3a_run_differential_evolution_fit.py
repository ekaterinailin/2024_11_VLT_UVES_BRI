
import sys
from scipy.optimize import differential_evolution
from funcs.lsr import setup_lsr_factory, get_lsr_data, register_lsr_models


if __name__ == "__main__":

    model_idn = int(sys.argv[1])

    model, vmids, alphas = setup_lsr_factory()

    model_funcs, model_funcs_pretty_names = register_lsr_models(model)

    path = "/home/ilin/Documents/2025_10_pydoppler/scripts/lsr1835bins20/"
    # path = "data/lsr1835bins20/"

    data, data_err = get_lsr_data(path, vmids)

    model_func = model_funcs[model_idn]

    param_names = model.get_param_names(model_func)
    print(f"Model: {model_func.__name__}, Parameters: {param_names}")

    
    fit_func = model.create_fitness_function(model_func, data)

    bounds = model.get_bounds(model_func)
    print(f"Bounds for model {model_func.__name__}: {bounds}")

    result = differential_evolution(fit_func, bounds=bounds, popsize=10)
    print(f"Best-fit parameters for model {model_func.__name__}: {result.x}")
    print(f"Chi-squared: {result.fun}")

    save_to = f"results_lsr/{model_func.__name__}_fit_results.json"

    model.save_fit_result(
        model_func=model_func,
        best_params=result.x,
        chi2=result.fun,
        filepath=save_to,
        metadata={
            'data_file': 'observations.fits',
            'notes': 'Initial fit with default bounds'
        },
        optimizer_info={
            'success': result.success,
            'nfev': result.nfev,
            'message': result.message
        }
    )


    # Load and inspect
    results = model.load_fit_result(save_to)
    print(f"Best chi2: {results['chi2']}")
    print(f"Parameters: {results['parameters']}")
    print(40*"-+" + "\n\n\n")

    # Print formatted summary
    model.print_fit_summary(save_to)

