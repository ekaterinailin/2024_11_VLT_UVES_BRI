import numpy as np
from funcs.fitting import model_spot, model_ring
import inspect
import json
from datetime import datetime
from pathlib import Path
from functools import wraps
import scipy
import inspect


class SpectralModelFactory:
    """Factory for creating spot and ring spectral 
    models with shared configuration.
    
    """
    
    def __init__(self, vbins, vmids, broaden, i_rot, omega, 
                 vmax, R_star, ddv, alphas, width1, ringwidth,
                 registry_file='model_registry.json', **kwargs):
        self.vbins = vbins
        self.vmids = vmids
        self.broaden = broaden
        self.width1 = width1
        self.ringwidth = ringwidth

        self.registry_file = Path(registry_file)
        
        self._common_kwargs = {
            'i_rot': i_rot,
            'omega': omega,
            'vmax': vmax,
            'R_star': R_star,
            'ddv': ddv,
            'alphas': alphas,
            'foreshortening': False,
            'obj_only': False
        }

        # add any additional kwargs to common kwargs
        self._common_kwargs.update(kwargs)
        
        # Define parameter bounds for all possible parameters
        self.parameter_bounds = {
            # Longitude parameters (0 to 2*np.pi degrees)
            'lon1': (-np.pi/4, np.pi*3/4),
            'lon2': (np.pi*3/4, 2*np.pi),
            'lon3': (0, 2*np.pi),
            'lon4': (0, 2*np.pi),

            # width1 and width2 (0 to 45 degrees)
            'width1': (0, np.pi/4),
            'width2': (0, np.pi/4),
            
            # Latitude parameters (0 to 90 degrees)
            'lat1': (0, np.pi/2),
            'lat2': (0, np.pi/2),
            'lat3': (0, np.pi/2),
            'lat4': (0, np.pi/2),
            'truelat': (0,np.pi),
            'truelat2': (0,np.pi),
            "trueringlat": (-np.pi/2,np.pi/2),
            
            # Amplitude parameters (0 to 1, or adjust as needed)
            'amplon1': (0,3),
            'amplon2': (0,3),
            'amplon3': (0,3),
            'amplon4': (0,3),
            'amplring': (0,3),
            'amplring1': (0,3),
            'amplring2': (0,3),
            
            # Ring-specific parameters
            'ringlat': (0, np.pi/2),
            'ringwidth': (0, np.pi/2),
            'ringwidth2': (0, np.pi/2),
            'i_mag': (0, np.pi/2),
            
            'alpha0': (0, 2*np.pi),
            'alpha_0': (0, 2*np.pi), 
            'amplback': (0,3),
        }

        # Registry to store model functions by name
        self._model_registry = {}

        # Load existing registry if available
        self._load_registry()
    
    def spot(self, lat, lon, width, ampl):
        """Create a spot spectrum."""
        return model_spot(
            self.vbins, self.vmids, lat, lon, width, ampl, self.broaden,
            typ="spot", **self._common_kwargs
        )
    
    def ring(self, i_mag, phimax, dphi, alpha_0, ampl):
        """Create a ring spectrum."""
        return model_ring(
            self.vbins, self.vmids, i_mag, phimax, dphi, alpha_0, 
            self.broaden, ampl, typ="ring", **self._common_kwargs
        )
    
    def equatorial_ring(self, amplback):
        """Create a standard equatorial ring."""
        return self.ring(0, self.ringwidth/2, self.ringwidth, 0, amplback)
    
    # @staticmethod
    def combine(self, *components):
        """Combine spectral components with automatic normalization."""
        if self._common_kwargs["obj_only"]:
            return components
        else:
            return sum(components) - (len(components) - 1)
    
    @staticmethod
    def get_param_names(func):
        """Extract parameter names from a function."""
        sig = inspect.signature(func)
        return list(sig.parameters.keys())
    
    def get_bounds(self, model_func):
        """
        Get parameter bounds for a model function in the correct order.
        
        Parameters:
        -----------
        model_func : callable
            The model function
            
        Returns:
        --------
        list of tuples
            Bounds for each parameter in the order they appear in the function signature
            
        Example:
        --------
        >>> bounds = model.get_bounds(ring_only)
        >>> # Returns: [(0, 1), (0, np.pi), (1, 50), (0, 90), (0, 2*np.pi)]
        >>> # For parameters: [amplring, ringlat, ringwidth, i_mag, alpha0]
        """
        param_names = self.get_param_names(model_func)
        bounds = []
        
        for name in param_names:
            if name in self.parameter_bounds:
                bounds.append(self.parameter_bounds[name])
            else:
                raise ValueError(
                    f"No bounds defined for parameter '{name}'. "
                    f"Available parameters: {list(self.parameter_bounds.keys())}"
                )
        
        return bounds
    
    def set_bounds(self, param_name, bounds):
        """
        Set or update bounds for a specific parameter.
        
        Parameters:
        -----------
        param_name : str
            Name of the parameter
        bounds : tuple
            (min, max) bounds for the parameter
        """
        self.parameter_bounds[param_name] = bounds
    
    def update_bounds(self, bounds_dict):
        """
        Update multiple parameter bounds at once.
        
        Parameters:
        -----------
        bounds_dict : dict
            Dictionary of {param_name: (min, max)} bounds
        """
        self.parameter_bounds.update(bounds_dict)
    
    def create_fitness_function(self, model_func, data, penalty=1000):
        """
        Create a chi-squared fitness function for a given model.
        
        Parameters:
        -----------
        model_func : callable
            The model function to fit
        data : array-like
            The observed data to fit against
        penalty : float, optional
            Penalty value to return on error (default: 1000)
            
        Returns:
        --------
        callable
            Fitness function that takes a parameter array and returns chi-squared
        """
        param_names = self.get_param_names(model_func)
        n_params = len(param_names)
        
        def fitness(params):
            if len(params) != n_params:
                raise ValueError(
                    f"Expected {n_params} parameters {param_names}, got {len(params)}"
                )
            
            try:
                model_spectra = model_func(*params)
                chi2 = np.sum((data - model_spectra)**2 / model_spectra)
                return chi2 if np.isfinite(chi2) else penalty
            except Exception:
                return penalty
        
        fitness.param_names = param_names
        fitness.n_params = n_params
        fitness.model_func = model_func
        
        return fitness
    
    def get_bounds_dict(self, model_func):
        """
        Get bounds as a dictionary mapping parameter names to bounds.
        
        Parameters:
        -----------
        model_func : callable
            The model function
            
        Returns:
        --------
        dict
            Dictionary of {param_name: (min, max)}
        """
        param_names = self.get_param_names(model_func)
        return {name: self.parameter_bounds[name] for name in param_names}


    def register_model(self, name, func):
        """
        Register a model function for later retrieval.
        
        Parameters:
        -----------
        name : str
            Name to register the model under
        func : callable
            The model function
        """
        self._model_registry[name] = func
    
    def get_model(self, name):
        """
        Retrieve a registered model function by name.
        
        Parameters:
        -----------
        name : str
            Name of the registered model
            
        Returns:
        --------
        callable
            The model function
        """
        if name not in self._model_registry:
            raise ValueError(
                f"Model '{name}' not registered. "
                f"Available models: {list(self._model_registry.keys())}"
            )
        return self._model_registry[name]
    
    def save_fit_result(self, model_func, best_params, chi2, filepath, 
                       metadata=None, optimizer_info=None):
        """
        Save optimization results to a JSON file.
        
        Parameters:
        -----------
        model_func : callable or str
            The model function (or its registered name)
        best_params : array-like
            Best-fit parameter values
        chi2 : float
            Chi-squared value for the best fit
        filepath : str or Path
            Path to save the results
        metadata : dict, optional
            Additional metadata to save (e.g., data info, notes)
        optimizer_info : dict, optional
            Additional optimizer information (e.g., success, nfev, message)
            
        Example:
        --------
        >>> result = differential_evolution(fitness, bounds)
        >>> model.save_fit_result(
        ...     ring_only, 
        ...     result.x, 
        ...     result.fun,
        ...     'fit_results.json',
        ...     optimizer_info={'success': result.success, 'nfev': result.nfev}
        ... )
        """
        # Get model name
        if isinstance(model_func, str):
            model_name = model_func
            model_func = self.get_model(model_name)
        else:
            model_name = model_func.__name__
        
        # Get parameter names
        param_names = self.get_param_names(model_func)
        
        # Ensure best_params is the right length
        if len(best_params) != len(param_names):
            raise ValueError(
                f"Length mismatch: {len(best_params)} values for {len(param_names)} parameters"
            )
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(best_params, np.ndarray):
            best_params = best_params.tolist()
        
        # Create result dictionary
        result_dict = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                name: float(value) for name, value in zip(param_names, best_params)
            },
            'parameter_order': param_names,
            'best_params': [float(v) for v in best_params],
            'chi2': float(chi2),
            'n_params': len(param_names),
        }
        
        # Add optional information
        if metadata is not None:
            result_dict['metadata'] = metadata
        
        if optimizer_info is not None:
            result_dict['optimizer_info'] = optimizer_info
        
        # Save to JSON
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return result_dict
    
    def load_fit_result(self, filepath):
        """
        Load optimization results from a JSON file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the saved results file
            
        Returns:
        --------
        dict
            Dictionary with all saved information
            
        Example:
        --------
        >>> results = model.load_fit_result('fit_results.json')
        >>> print(results['parameters'])
        >>> print(results['chi2'])
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            result_dict = json.load(f)
        
        print(f"Loaded results from {filepath}")
        print(f"Model: {result_dict['model_name']}")
        print(f"Chi-squared: {result_dict['chi2']:.4f}")
        print(f"Timestamp: {result_dict['timestamp']}")
        
        return result_dict
    
    def reproduce_fit(self, filepath_or_dict):
        """
        Reproduce the best-fit model spectrum from saved results.
        
        Parameters:
        -----------
        filepath_or_dict : str, Path, or dict
            Either path to saved results file or loaded results dictionary
            
        Returns:
        --------
        array-like
            The model spectrum evaluated at best-fit parameters
            
        Example:
        --------
        >>> model_spectrum = model.reproduce_fit('fit_results.json')
        >>> # Or:
        >>> results = model.load_fit_result('fit_results.json')
        >>> model_spectrum = model.reproduce_fit(results)
        """
        # Load results if filepath provided
        if isinstance(filepath_or_dict, (str, Path)):
            results = self.load_fit_result(filepath_or_dict)
        else:
            results = filepath_or_dict
        
        # Get model function
        model_name = results['model_name']
        model_func = self.get_model(model_name)
        
        # Get best parameters
        best_params = results['best_params']
        
        # Compute and return model spectrum
        model_spectrum = model_func(*best_params)
        
        print(f"Reproduced model '{model_name}' with {len(best_params)} parameters")
        
        return model_spectrum
    
    def print_fit_summary(self, filepath_or_dict):
        """
        Print a formatted summary of fit results.
        
        Parameters:
        -----------
        filepath_or_dict : str, Path, or dict
            Either path to saved results file or loaded results dictionary
        """
        # Load results if filepath provided
        if isinstance(filepath_or_dict, (str, Path)):
            results = self.load_fit_result(filepath_or_dict)
        else:
            results = filepath_or_dict
        
        print("\n" + "="*60)
        print(f"FIT RESULTS: {results['model_name']}")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Chi-squared: {results['chi2']:.6f}")
        print(f"\nBest-fit parameters:")
        print("-"*60)
        
        for name, value in results['parameters'].items():
            print(f"  {name:15s}: {value:12.6f}")
        
        if 'optimizer_info' in results:
            print(f"\nOptimizer information:")
            print("-"*60)
            for key, value in results['optimizer_info'].items():
                print(f"  {key:15s}: {value}")
        
        if 'metadata' in results:
            print(f"\nMetadata:")
            print("-"*60)
            for key, value in results['metadata'].items():
                print(f"  {key:15s}: {value}")
        
        print("="*60 + "\n")

    def register(self, func=None, name=None):
        """
        Decorator to automatically register a model function.
        
        Can be used as:
            @model.register
            def my_model(...):
                ...
        
        Or with custom name:
            @model.register(name='custom_name')
            def my_model(...):
                ...
        
        Parameters:
        -----------
        func : callable, optional
            The function to register (when used without parentheses)
        name : str, optional
            Custom name for the model (defaults to function name)
        """
        def decorator(f):
            model_name = name if name is not None else f.__name__
            
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            
            # Register the function
            self._model_registry[model_name] = {
                'function': wrapper,
                'param_names': self.get_param_names(f),
                'docstring': f.__doc__,
                'source': inspect.getsource(f) if inspect.getsource else None
            }
            
            # Save updated registry
            self._save_registry()
            
            print(f"Registered model: '{model_name}' with parameters {self._model_registry[model_name]['param_names']}")
            
            return wrapper
        
        # Handle both @register and @register()
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _save_registry(self):
        """Save the model registry to file."""
        registry_data = {
            'timestamp': datetime.now().isoformat(),
            'models': {
                name: {
                    'param_names': info['param_names'],
                    'docstring': info['docstring'],
                    'source': info['source']
                }
                for name, info in self._model_registry.items()
            }
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _load_registry(self):
        """Load the model registry from file if it exists."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                print(f"Loaded model registry from {self.registry_file}")
                print(f"Available models: {list(registry_data['models'].keys())}")
                
                # Note: We can't restore the actual functions from the registry file
                # Functions must be re-registered by running the code that defines them
                
            except Exception as e:
                print(f"Warning: Could not load registry from {self.registry_file}: {e}")
    
    def get_model(self, name):
        """
        Retrieve a registered model function by name.
        
        Parameters:
        -----------
        name : str
            Name of the registered model
            
        Returns:
        --------
        callable
            The model function
        """
        if name not in self._model_registry:
            raise ValueError(
                f"Model '{name}' not registered. "
                f"Available models: {list(self._model_registry.keys())}\n"
                f"Make sure to define and register the model before loading results."
            )
        return self._model_registry[name]['function']
    
    def list_models(self):
        """List all registered models."""
        if not self._model_registry:
            print("No models registered yet.")
            return []
        
        print("\nRegistered Models:")
        print("-" * 60)
        for name, info in self._model_registry.items():
            params = ', '.join(info['param_names'])
            print(f"  {name}: ({params})")
            if info['docstring']:
                print(f"    {info['docstring'].strip()}")
        print("-" * 60)
        
        return list(self._model_registry.keys())
    
    def create_ultranest_prior(self, model_func):
        """
        Create prior transformation for ultranest (unit cube to physical parameters).
        
        Handles special transformations:
        - Latitude parameters (lat1, lat2, etc.): uniform in sin(lat)
        - i_mag: uniform in sin(i_mag)
        - ringlat: uniform in cos(ringlat)
        - All other parameters: uniform in bounds
        
        Parameters:
        -----------
        model_func : callable
            The model function
            
        Returns:
        --------
        callable
            Prior transformation function
        """
        param_names = self.get_param_names(model_func)
        bounds = self.get_bounds(model_func)
        n_params = len(param_names)
        # cosine_distribution = scipy.stats.cosine()
        
        def prior_transform(cube):
            """Transform unit cube to physical parameters."""
            params = np.zeros(n_params)
            
            for i, (name, (lower, upper)) in enumerate(zip(param_names, bounds)):
                
                if ('lat' in name) | (name=='ringlat') | (name=='i_mag') | (name=='truei_mag') | (name=='trueringlat'):
                    params[i] = cube[i]
                elif "truelat" in name:
                    params[i] = cube[i]  # cosine distribution between -pi/2 and pi/2
                else:
                    # All other parameters: uniform in bounds
                    params[i] = cube[i] * (upper - lower) + lower
            return params
        
        return prior_transform
    
    def create_ultranest_likelihood(self, model_func, data, data_err):
        """
        Create log-likelihood function for ultranest.
        
        Parameters:
        -----------
        model_func : callable
            The model function
        data : array-like
            Observed data
        data_err : array-like
            Uncertainties on the data
            
        Returns:
        --------
        callable
            Log-likelihood function
        """
        yerr2 = data_err ** 2
        
        def log_likelihood(params):
            try:

                # get through param names and convert to sin(lat) or cos(ringlat) if needed
                for i, name in enumerate(self.get_param_names(model_func)):
                    if name == 'lat1' or name == 'lat2' or name == 'lat3' or name == 'lat4':
                        params[i] = np.arcsin(params[i])
                    elif name == 'ringlat':
                        params[i] = np.arccos(params[i])
                    elif name == "i_mag":   
                        params[i] = np.arcsin(params[i])
                    elif name == 'truei_mag':
                        params[i] = np.arcsin(params[i]) + np.pi/2
                    elif name == 'trueringlat':
                        params[i] = np.arccos(params[i])
                    elif name == 'truelat':
                        params[i] = np.arcsin(params[i]) + np.pi/2  # convert back to 0 to pi
                    elif name == 'truelat2':
                        params[i] = np.arcsin(params[i]) + np.pi/2  # convert back to 0 to pi
                    elif name == "ringwidth":
                        params[i] = params[i]  # sign flip to go to colatitude
                model_spectra = model_func(*params)
                logf = -0.5 * np.sum(np.log(2 * np.pi * yerr2))
                loglike = -0.5 * np.sum((data - model_spectra) ** 2 / yerr2) + logf
                return loglike if np.isfinite(loglike) else -1e100
            except Exception:
                return -1e100
        
        return log_likelihood
    
    def get_wrapped_params(self, model_func):
        """
        Determine which parameters are periodic (wrapped).
        
        Parameters:
        -----------
        model_func : callable
            The model function
            
        Returns:
        --------
        list of bool
            True for wrapped parameters, False otherwise
        """
        param_names = self.get_param_names(model_func)
        
        wrapped = []
        for name in param_names:
            # Longitude and angle parameters are periodic
            is_wrapped = (name.startswith('lon') or 
                         name.startswith('alpha') or
                         name == 'alpha_0')
            wrapped.append(is_wrapped)
        
        return wrapped





