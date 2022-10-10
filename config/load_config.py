import os
import yaml
import functools
from argparse import Namespace
from collections.abc import MutableMapping

CONFIG_FILES_DIR = "config_files"

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# From: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        try:
            return getattr(obj, attr, *args)
        except AttributeError:
            setattr(obj, attr, Namespace())
            return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def is_yaml_file(x):
    if isinstance(x, str):
        return x.endswith("yaml") or x.endswith("yml")
    return False


def get_repo_rootdir():
    import shlex
    from subprocess import check_output
    repo_rootdir = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')
    return repo_rootdir


def unfold_config(token, no_unfolding_for=[]):
    '''
    Parameters: 
      token: a recursive structure composed of either 1. a path to a yaml file or 2. a dictionary composed of such structures.
      no_unfolding_for: a list of dict keys for which the yaml shouldn't be unfolded, and instead kept as a path
    Returns: A dictionary with all the yaml files replaces by their content.
    '''

    #
    if is_yaml_file(token):
        #TODO: COMMENT AND DOCUMENT THIS!!!
        yaml_file_base = token
        try:            
            yaml_dir = get_repo_rootdir()
            yaml_file = os.path.join(yaml_dir, yaml_file_base)
            token = yaml.safe_load(open(yaml_file))
        except FileNotFoundError:
            yaml_dir = os.path.join(get_repo_rootdir(), CONFIG_FILES_DIR)
            yaml_file = os.path.join(yaml_dir, yaml_file_base)
            token = yaml.safe_load(open(yaml_file))

    if isinstance(token, dict):
        for k, v in token.items():
            if k not in no_unfolding_for:
                token[k] = unfold_config(v, no_unfolding_for)

    return token


def to_dict(token):
    '''
    Converts a (possibly nested) namespace to a nested dictionary
    '''

    if isinstance(token, Namespace):
        namespace_as_dict = token.__dict__
        token = {k: to_dict(v) for k, v in namespace_as_dict.items()}
    return token


def recursive_namespace(dd):
    '''
    Converts a (possibly nested) dictionary into a namespace.
    This allows for auto-completion.
    '''
    for d in dd:
        has_any_dicts = False
        if isinstance(dd[d], dict):
            dd[d] = recursive_namespace(dd[d])
            has_any_dicts = True
    return Namespace(**dd)


def sanity_check(config):
    
    '''
    Perform sanity check on the provided configuration.
    '''

    pol_deg_dim = len(config.network_architecture.convolution.parameters.polynomial_degree)
    downsampling_factors_dim = len(config.network_architecture.pooling.parameters.downsampling_factors)
    
    try:
      n_channels_dim = len(config.network_architecture.convolution.channels)
    except:
      n_channels_dim = len(config.network_architecture.convolution.channels_enc)

    if not ((pol_deg_dim == downsampling_factors_dim) and (pol_deg_dim == n_channels_dim)):       
       raise ValueError(
          f"Dimensions of polynomial degrees, downsampling factors and number of channels should match \
          (but are {pol_deg_dim}, {downsampling_factors_dim} and {n_channels_dim}.)"
       )


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    return dict(_flatten_dict_gen(d, parent_key, sep))


def load_yaml_config(yaml_config_file):

    # config = yaml.safe_load(config)    
    config = unfold_config(yaml_config_file)    
        # I am using a namespace instead of a dictionary mainly because it enables auto-completion
    config = recursive_namespace(config)

    # The following parameters are meant to be lists of numbers, so they are parsed here from their string representation in the YAML file.
    if isinstance(config.network_architecture.convolution.parameters.polynomial_degree, str):
      config.network_architecture.convolution.parameters.polynomial_degree = \
      [int(x) for x in config.network_architecture.convolution.parameters.polynomial_degree.split()]
    
    if isinstance(config.network_architecture.pooling.parameters.downsampling_factors, str):
      config.network_architecture.pooling.parameters.downsampling_factors = \
      [int(x) for x in config.network_architecture.pooling.parameters.downsampling_factors.split()]

    if hasattr(config.network_architecture.convolution, "channels"):
      if isinstance(config.network_architecture.convolution.channels, str):
        config.network_architecture.convolution.channels = \
        [int(x) for x in config.network_architecture.convolution.channels.split()]      
        
    if hasattr(config.network_architecture.convolution, "channels_enc"):
      if isinstance(config.network_architecture.convolution.channels_enc, str):
        config.network_architecture.convolution.channels_enc = \
        [int(x) for x in config.network_architecture.convolution.channels_enc.split()]
  
    if hasattr(config.network_architecture.convolution, "channels_dec"):
      if isinstance(config.network_architecture.convolution.channels_dec, str):
        config.network_architecture.convolution.channels_dec = \
        [int(x) for x in config.network_architecture.convolution.channels_dec.split()]

    if hasattr(config.network_architecture, "activation_function"):
      if isinstance(config.network_architecture.activation_function, str):
        config.network_architecture.activation_function = \
        [x for x in config.network_architecture.activation_function.split()]
    
    sanity_check(config)

    return config
