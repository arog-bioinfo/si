import numpy as np
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
    # Validate the parameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    results = {'scores': [], 'hyperparameters': []}

    # Generate all possible combinations
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())

    # Perform n_iter random searches
    for _ in range(n_iter):
        # Generate a random combination
        combination = []
        for values in param_values:
            combination.append(np.random.choice(values))

        # Parameter configuration
        parameters = {}

        # Set the parameters
        for parameter, value in zip(param_names, combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # Cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Add the mean score
        results['scores'].append(np.mean(score))

        # Add the hyperparameters
        results['hyperparameters'].append(parameters)

    # Find the best score and hyperparameters
    best_idx = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_idx]
    results['best_score'] = results['scores'][best_idx]

    return results
