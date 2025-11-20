# pmm
A toolkit for running a parmetric matrix model (PMM) on eigenvalue data read from a file.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KSHobbyProjs/pmm.git
cd pmm
pip install -r requirements.txt
```
Dependencies include `numpy`, `scipy`, `h5py`, `jax`

---

## Usage
Run a PMM using `python run_pmm.py`.  

```bash
python run_pmm.py \
      input_file.h5 \
    --config-file config.txt \
    --parameters 5.0,20.0:50 \
    --epochs 10000 \
    --save-energies pmm_predictions.h5
```

---

## Input

The program requires an input file containing parameters and eigenvalue data.
- If the input filename ends with `.h5`, the program expects a full HDF5 dataset.  
- If the input filename ends with any other file extension, the program expects a `.dat`-style column-format summary.

- ### `.h5` files — full dataset:
  - `parameters` — 1D NumPy array of parameter values (`len(parameters)`)  
  - `energies` — 2D NumPy array of shape (`len(parameters)`, `knum`)  
    (*knum* is the number of eigenpairs per parameter, as set by `--knum`)  
  - `eigenvectors` — 3D NumPy array of shape (`len(parameters)`, `knum`, `vector dimension`)

- ## `.dat` files — summary dataset:
  - Column 1: `parameters` — parameter values  
  - Columns 2..(knum+1): `energies` — one column per eigenvalue  
  - Note: `eigenvectors` are **not** included

### Notes on Data Shapes

- Preferred shapes: `parameters` as a 1D array, and `energies` as 2D `(len(parameters), knum)` array.  
- The program will attempt to **infer the correct structure** if the input does not exactly match this layout.

---

## Output

The program writes results to a file specified by the user.  
- If the output filename ends with `.h5`, the program writes a full HDF5 dataset.  
- Any other file extension will produce a `.dat`-style column-format summary.

### `.h5` files — full dataset
- `parameters` — 1D NumPy array of parameter values (`len(parameters)`)  
- `energies` — 2D NumPy array of shape (`len(parameters)`, `knum`) containing eigenvalues  
  (*knum* is the number of eigenpairs per parameter, as set by `--knum`)  
- `eigenvectors` — 3D NumPy array of shape (`len(parameters)`, `knum`, `vector dimension`) containing eigenvectors

### `.dat` files — summary dataset
- Column 1: `parameters` — parameter values  
- Columns 2..(knum+1): `energies` — one column per eigenvalue  
- Note: `eigenvectors` are **not** included in `.dat` files

---

## Key Arguments
- `input_file`          : Input file with eigenvalue data.
- `--pmm-name`          : Name of the PMM type to run.
- `--parameters`        : Parameter values in `start,end:len`, `start,end:len,exp` or `val1,val2,val3` format.
  Example: `5.0,6.0,7.0` or `5.0,20.0:150`.
- `--config-file`       : Path to a config file that overrides default PMM parameters.
- `--config`            : Comma-separated key=val pairs to override default PMM parameters (takes precendence over loaded config file if given).
  Example: `eta=1.0e-2,num_primary=5`.
- `--epochs`            : Number of cycles to train PMM.
- `--save-energies`     : Output filename for predicted energies.
- `--verbose`           : Increase verbosity.

---

## Adding a PMM Type:
PMM types are stored in `src/pmm.py`. To add a new PMM type, just add a class in this module that subclasses `PMM` and
- Modify `loss(params, Ls, energies, l2)` to adjust how the loss is computed (default is mean squared error between predicted eigenvalues and sampled eigenvalues).
Once defined, the model can be used with `--pmm-name [NewPMMClassName]`.
