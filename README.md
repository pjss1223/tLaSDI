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
- Couette flow of an Oldroyd-B fluid
- Two gas containers exchanging heat and volume
- 1D Burgers’ equation

The data for all examples will be made available on request.

## Description of Arguments

- Autoencoder architecture
  
| Argument | Description | Choices |
| -------- | -------- | -------- |
| `--activation_AE`   | str, activation function for AE   |  `tanh`, `relu`, `linear`, `sin`, `gelu`, `elu`, `silu` |
| `--AE_width1`       | int, width of the first layer of AE | |
| `--AE_width1`       | int, width of the first layer of AE | |
| `latent_dim`        | int, latent space dimension | |

- DI model architecture

| Argument | Description | Choices |
| -------- | -------- | -------- |
| `--net`  | str, DI model choice | `GFINNs`, `SPNN` | 
| `--activation` | str, activation function for DI model  | `tanh`, `relu`, `linear`, `sin`, `gelu`, `elu`, `silu`  |
| `--layers` | int, number of layers in DI model | |
| `--width` | int, width of DI model | |
| `--extraD_L` | int, # of skew-symmetric matrices generated to construct L | |
| `--extraD_M` | int, # of skew-symmetric matrices generated to construct M | |

- Hypernetwork architecture (parametric case)

| Argument | Description | Choices |
| -------- | -------- | -------- |
|`--act_hyper` | int, activation function of hypernetwork | `tanh`, `relu`, `linear`, `sin`, `gelu`, `elu`, `silu`  |
|`--depth_hyper` | int, depth of hypernetwork | |
|`--width_hyper` | int, width of hypernetwork | |


- General
  
| Argument | Description | Choices |
| -------- | -------- | -------- |
|`--load_model`| str2bool, load previously trained model | Default: `False`|
|`--iterations`| int, number of iterations | |
|`--load_iterations`| int, previous number of iterations for loaded networks | |
|`--lambda_r_AE`| float, penalty for reconstruction loss | Default: `1e-2`|
|`--lambda_jac_AE`|float, penalty for Jacobian loss | Default: `1e-2`|
|`--lambda_dx` | float, penalty for consistency part of model loss | Default: `1e-7`|
|`--lambda_dz` | float, penalty for model approximation part of model loss | Default: `1e-7`|
|`--lambda_deg` | float, penalty for degeneracy loss (for SPNN) | Default: 1e-3|
|`--order` | int, DI model time integrator 1:Euler, 2:RK23, 4:RK45 | `1`, `2`, `4`|
|`--update_epochs` | int, greedy sampling frequency (parametric case) | Default: `1000`| 

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


## Acknowledgements

This code makes the use of the codes from the following projects:

- Hernández, Quercus and Badías, Alberto and González, David and Chinesta, Francisco and Cueto, Elías. "Deep learning of thermodynamics-aware reduced-order models from data." Computer Methods in Applied Mechanics and Engineering (2021)
- Zhang, Zhen and Shin, Yeonjong and Karniadakis George. "GFINNs: GENERIC formalism informed neural networks for deterministic and stochastic dynamical systems" (2022)
- He, Xiaolong and Choi, Youngsoo and Fries, William and Belof, Jon and Chen, Jiun-Shyan. "gLaSDI: Parametric Physics-informed Greedy Latent Space Dynamics Identiﬁcation" (2023)

We are thankful to the authors for providing the codes.

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
