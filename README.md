## Title

tLaSDI: Thermodynamics-informed latent space dynamics identification


## Abstract 

We propose a latent space dynamics identification method, namely tLaSDI, that embeds the first and second principles of thermodynamics. 
The latent variables are learned through an autoencoder as a nonlinear dimension reduction model.
The latent dynamics are constructed by a neural network-based model that precisely preserves certain structures for the thermodynamic laws through the GENERIC formalism.
An abstract error estimate is established, which provides a new loss formulation involving the Jacobian computation of autoencoder.
The autoencoder and the latent dynamics are simultaneously trained to minimize the new loss.
Computational examples demonstrate the effectiveness of tLaSDI, which exhibits robust generalization ability, even in extrapolation.
In addition, an intriguing correlation is empirically observed between a quantity from tLaSDI in the latent space and the behaviors of the full-state solution.

## Required Packages

- Python: 3.9.16
- PyTorch: 1.13.0
- Numpy: 1.23.4
- Scipy: 1.8.1
- Sklearn: 1.2.2
- Matplotlib: 3.7.1
- Seaborn: 0.13.2
- cuda: 11.6.1

## Examples

Three examples are provided, including
- Couette flow of an Oldroyd-B fluid (VC)
- Two gas containers exchanging heat and volume (GC)
- 1D Burgers’ equation (1DBG)

The data for all examples will be made available on request.

## Description of Arguments

- Autoencoder architecture
  
| Argument | Description | Choices |
| -------- | -------- | -------- |
| `--activation_AE`   | `str`, activation function for AE   |  `tanh`, `relu`, `linear`, `sin`, `gelu`, `elu`, `silu` |
| `--AE_width1`       | `int`, width of the first layer of AE | Default: `160` |
| `--AE_width1`       | `int`, width of the first layer of AE | Default: `160` |
| `latent_dim`        | `int`, latent space dimension | Default: `8` |

- DI model architecture

| Argument | Description | Choices |
| -------- | -------- | -------- |
| `--net`  | `str`, DI model choice | `GFINNs`, `SPNN` | 
| `--activation` | `str`, activation function for DI model  | `tanh`, `relu`, `linear`, `sin`, `gelu`, `elu`, `silu`  |
| `--layers` | `int`, number of layers in DI model | Default: `5` |
| `--width` | `int`, width of DI model | Default: `100` |
| `--extraD_L` | `int`, # of skew-symmetric matrices generated to construct L | Default: `8` |
| `--extraD_M` | `int`, # of skew-symmetric matrices generated to construct M | Default: `8` |

- Hypernetwork architecture (parametric case)

| Argument | Description | Choices |
| -------- | -------- | -------- |
|`--act_hyper` | `int`, activation function of hypernetwork | `tanh`, `relu`, `linear`, `sin`, `gelu`, `elu`, `silu`  |
|`--depth_hyper` | `int`, depth of hypernetwork | Default: `3`  |
|`--width_hyper` | `int`, width of hypernetwork | Default: `20` |


- General
  
| Argument | Description | Choices |
| -------- | -------- | -------- |
|`--load_model`| `str2bool`, load previously trained model | Default: `False`|
|`--iterations`| `int`, number of iterations | Default: `40000`|
|`--load_iterations`| `int`, previous number of iterations for loaded networks | Default:`0` |
|`--lambda_r_AE`| `float`, penalty for reconstruction loss | Default: `1e-1`|
|`--lambda_jac_AE`|`float`, penalty for Jacobian loss | Default: `1e-2`|
|`--lambda_dx` | `float`, penalty for consistency part of model loss | Default: `1e-8`|
|`--lambda_dz` | `float`, penalty for model approximation part of model loss | Default: `1e-8`|
|`--lambda_deg` | `float`, penalty for degeneracy loss (for SPNN) | Default: `1e-3`|
|`--order` | `int`, DI model time integrator 1:Euler, 2:RK23, 4:RK45 | `1`, `2`, `4`|
|`--update_epochs` | `int`, greedy sampling frequency (parametric case) | Default: `1000`| 
|`--miles_lr` | `int`, learning rate decay frequency | Default: `1000`|
|`--gamma_lr` | `float`, rate of learning rate decay | Default: `.99`|

## How to run the examples

- Couette flow of an Oldroyd-B fluid
  
```python
python main_VC_tLaSDI.py --lambda_r_AE 1e-1 --lambda_jac_AE 1e-2 --lambda_dx 1e-8 --lambda_Dz 1e-8 ...
```

- Two gas containers exchanging heat and volume
  
```python
python main_GC_tLaSDI.py --lambda_r_AE 1e-1 --lambda_jac_AE 1e-2 --lambda_dx 1e-7 --lambda_Dz 1e-7 ...
```

- 1D Burgers’ equation
  
```python
python main_1DBG_tLaSDI_param.py --lambda_r_AE 1e-1 --lambda_jac_AE 1e-9 --lambda_dx 1e-7 --lambda_Dz 1e-7 ...
```

## Authors

- Jun Sur Richard Park (Korea Institue For Advanced Study)
- Siu Wun Cheung (Lawrence Livermore National Laboratory)
- Youngsoo Choi (Lawrence Livermore National Laboratory)
- Yeonjong Shin (North Carolina State University)

## Acknowledgements

This code makes the use of the codes from the following projects:

- Hernández, Quercus and Badías, Alberto and González, David and Chinesta, Francisco and Cueto, Elías. "Deep learning of thermodynamics-aware reduced-order models from data." Computer Methods in Applied Mechanics and Engineering (2021)
- Zhang, Zhen and Shin, Yeonjong and Karniadakis George. "GFINNs: GENERIC formalism informed neural networks for deterministic and stochastic dynamical systems" (2022)
- He, Xiaolong and Choi, Youngsoo and Fries, William and Belof, Jon and Chen, Jiun-Shyan. "gLaSDI: Parametric Physics-informed Greedy Latent Space Dynamics Identiﬁcation" (2023)

We are thankful to the authors for providing the codes.

J. S. R. Park was partially supported by Miwon Du-Myeong
Fellowship via Miwon Commercial Co., Ltd. and a KIAS Individual Grant (AP095601)
via the Center for AI and Natural Sciences at Korea Institute for Advanced Study.
J. S. R. Park would like to thank Dr. Quercus Hernandez and Zhen Zhang for their
helpful guidance on the implementation of TA-ROM and GFINNs. S. W. Cheung and
Y. Choi were partially supported for this work by Laboratory Directed Research and
Development (LDRD) Program by the U.S. Department of Energy (24-ERD-035). Y.
Choi was partially supported for this work by the U.S. Department of Energy, Office
of Science, Office of Advanced Scientific Computing Research, as part of the CHaRM-
NET Mathematical Multifaceted Integrated Capability Center (MMICC) program,
under Award Number DE-SC0023164 at Lawrence Livermore National Laboratory.
Y. Shin was partially supported for this work by the NRF grant funded by the Min-
istry of Science and ICT of Korea (RS-2023-00219980).


## License
tLaSDI is distributed under the terms of the MIT license. All new contributions must be made under the MIT license. See
[LICENSE-MIT](https://github.com/pjss1223/tLaSDI/blob/main/LICENSE)

LLNL Release Number: LLNL-CODE-867909

## Citation

If you find this code useful, please cite the work as

```bibtex
@article{park2024tlasdi,
  title={tLaSDI: Thermodynamics-informed latent space dynamics identification},
  author={Park, Jun Sur Richard and Cheung, Siu Wun and Choi, Youngsoo and Shin, Yeonjong},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={429}
  pages={117144},
  year={2024}
  publisher={Elsevier}
}


