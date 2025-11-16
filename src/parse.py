#!/usr/bin/env python3

import numpy as np
from . import pmm

import logging
logger = logging.getLogger(__name__)

def parse_config(config, config_file):
    pass

def parse_pmm_string(pmm_string):
    """
    Parses a CLI argument for PMM initialization.

    Parameters
    ----------
    pmm_string : str
        CLI string containing the class name of the PMM type.

    Returns
    -------
    PMMClass : PMM
        A class object that subclasses `PMM`.

    Examples
    --------
    parse_pmm_instance("PMM")          -> pmm.PMM
    parse_pmm_instance("PMMParity")    -> pmm.PMMParity()
    """
    s = pmm_name.strip()
    try:
        return getattr(pmm, s)
    except AttributeError as e:
        raise RuntimeError(f"PMM {pmm_name} not found in `pmm` module.") from e
    
def _parse_kwargs(kwargs_string):
    """
    Parses a comma-separated list of key=val pairs into a dictionary.

    Parameters
    ----------
    kwargs_string : str
        Comma-separated string containing kwarg info.

    Returns
    -------
    kwargs : dict
        Dictionary containing the key=val pairs in `kwargs_string`.
    
    Example
    -------
    'N=32,V0=-4.0,R=2.0' -> {"N" : 32, "V0" : -4.0, "R" : 2.0}.
    """
    
    def _convert_value(v):
        """
        Takes a string and determines if it's meant to be an int, float, bool, or str
        """
        v = v.strip().lower()
        try:
            return int(v)       # check int
        except ValueError:
            pass

        try:
            return float(v)     # check float
        except ValueError:
            pass

        if v == "true":
            return True
        elif v == "false":
            return False        
        return v                # fallback string

    s = kwargs_string.strip()
    kwargs = {}
    for kv in s.split(","):
        if not kv.strip():
            continue
        if "=" not in kv:
            raise RuntimeError(f"Invalid argument input: '{kv}'. Kwarg arguments need to be input in the form 'key1=val1,key2=val2'")
        k, v = kv.split("=", 1)   
        kwargs[k.strip()] = _convert_value(v)
    return kwargs

def _parse_config_file():
    pass

