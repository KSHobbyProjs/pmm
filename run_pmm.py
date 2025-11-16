#!/usr/bin/env python3

import os, sys
import argparse
import time
import logging
logger = logging.getLogger(__name__)

import h5py
import numpy as np

from src import parse
from src import io
from src import logging_utils

def setup_logging(verbose=0):
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
            level = level,
            format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt = "%H:%M:%S"
        )

def main():
    parser = argparse.ArgumentParser(description="Run a Parametric Matrix Model using energy data loaded from a file.")
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("-p", "--pmm-name", type=str, default="PMM", help="Name of the PMM class to use.")
    parser.add_argument("-d", "--dim", type=int, default=10, help="The dimension of the PMM.")
    parser.add_argument("-c", "--config", type=str, default="", help="Comma-separated key=val pairs to override default PMM parameters. E.g., eta=1.0e-2,beta1=0.9.")
    parser.add_argument("--config-file", type=str, default=None, help="Path to a file to load PMM parameters. key=val pairs passed through -c overwrite config files.")
    parser.add_argument("-o", "--out", type=str, default=None, help="Path to file for energy data output.")
    parser.add_argument("-s", "--save", type=str, default=None, help="Path to file for saving the PMM state.")
    parser.add_argument("-e", "--epochs", type=int, default=10000, help="Number of cycles to run PMM algorithm for.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    parser.add_argument("-L", "--parameters", type=str, default="1.0", help="Parameter values at which to predict energies.")
    parser.add_argument("--store-loss", type=int, default=100, help="Frequency for storing the computing loss.")

    args = parser.parse_args()    # parse CLI args 
    setup_logging(args.verbose)   # setup logging
    start = time.time()           # start timer to measure elapsed time
    print(f"Input = {args.input_file}\tPMM = pmm.{args.pmm_name}\tdim = {args.dim}\tEpochs = {args.epochs}")

    # parse config
    logger.info(f"Parsing config data in config={args.config} and config_file={args.config_file}.")
    PMMClass = run.parse_pmm_string(args.pmm_name)
    pmm_kwargs = run.parse_config(args.config, args.config_file)
    predict_Ls = run.parse_parameter_values(args.parameters)
    logger.info(f"Finished parsing config data.")

    # instantiate and sample PMM with config data if any given
    logger.info(f"Instantiating pmm type pmm.{args.pmm_name} with config data.")
    pmm_instance = PMMClass(args.dim, **pmm_kwargs)

    # load data from input_file
    logger.info(f"Loading energy data from input file.")
    if args.input_file.endswith(".h5"):
        sample_Ls, sample_energies = load_energies_from_h5(args.input_file)
        logger.info(f"Loaded energy data from HDF5 file.")
    else:
        sample_Ls, sample_energies = load_energies_from_dat(args.input_file)
        logger.info(f"Loaded energy data from .dat-type file.")

    # sample pmm with loaded energies
    logger.info("Sampling pmm with energies loaded from file.")
    pmm_instance.sample_energies(sample_Ls, sample_energies)

    # train the pmm for epoch number of cycles
    logger.info(f"Training pmm for {args.epochs} cycles and storing loss every {args.store_loss} cycles.")
    _, losses = pmm_instance.train_pmm(args.epochs, args.store_loss)

    # predict energies at Ls
    logger.info(f"Predicting energies of trained PMM.")
    predict_energies = pmm_instance.predict_energies(predict_Ls)

    # save energy data, pmm state, and loss data if desired
    if args.out:
        logger.info(f"Saving energy data to out file.")
        if args.out.endswith(".h5"):
            save_energy_data_with_h5(args.out)
            logger.info(f"Saved energy to HDF5 file.")
        else:
            save_energy_data_with_dat(args.out)
            logger.info(f"Saved energy to .dat-type file.")

    if args.save:
        logger.info("Saving PMM state.")
        save_pmm_state(args.save)

    if args.loss:
        logger.info("Saving loss data.")
        save_loss(args.loss)

    # print data
    for i, L in enumerate(predict_Ls):
        print(f"Spectrum at L = {L:.3f}")
        print(f"\t{predicted_energies[i]}")
    
    end = time.time()
    # print total time elapsed
    print(f"Done.\nElapsed time: {end - start:.3f} seconds.")

if __init__=="__main__":
    main()
