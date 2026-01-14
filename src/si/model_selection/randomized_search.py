import numpy as np
from itertools import product
from typing import Callable, Dict, Any, List

from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

#Ex.11
def randomized_search_cv(model,
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, List[Any]],
                         scoring: Callable = None,
                         cv: int = 5,
                         n_iter: int = 10) -> Dict[str, Any]:
    """
    Performs a randomized search cross validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, List[Any]]
        The hyperparameter grid to sample from.
    scoring: Callable
        The scoring function to use.
    cv: int
        The number of cross validation folds.
    n_iter: int
        The number of parameter settings that are sampled.

    Returns
    -------
    results: Dict[str, Any]
        The results of the randomized search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # 1) Validate grid
    for param in hyperparameter_grid:
        if not hasattr(model, param):
            raise AttributeError(f"Model {model} does not have parameter {param}.")

    results = {'scores': [], 'hyperparameters': []}

    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())

    # All possible combinations (Cartesian product)
    all_combinations = list(product(*param_values))
    n_possible = len(all_combinations)

    # Sample n_iter combinations (without replacement)
    if n_iter >= n_possible:
        sampled_idx = np.arange(n_possible)
        np.random.shuffle(sampled_idx)
    else:
        sampled_idx = np.random.choice(n_possible, size=n_iter, replace=False)

    # Iterate sampled combos
    for idx in sampled_idx:
        combo = all_combinations[idx]

        parameters = {}
        for param, value in zip(param_names, combo):
            setattr(model, param, value)
            parameters[param] = value

        scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)
        results['scores'].append(float(np.mean(scores)))
        results['hyperparameters'].append(parameters)

    best_idx = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score'] = results['scores'][best_idx]

    return results
