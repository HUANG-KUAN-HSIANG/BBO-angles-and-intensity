import numpy as np
import matplotlib.pyplot as plt

# Constants
lambda_fund = 1.030  # fundamental wavelength in microns
lambda_shg = lambda_fund / 2
deg2rad = np.pi / 180

# Sellmeier equations for BBO
def n_o_BBO(wavelength):
    # wavelength in microns
    lam2 = wavelength**2
    return  np.sqrt(2.7366122 + (0.018572 / (lam2 - 0.0178746)) - 0.0143756 * lam2)

def n_e_BBO(wavelength):
    lam2 = wavelength**2
    return np.sqrt(2.3698703 + (0.0128445 / (lam2 - 0.0153064)) - 0.0029129 * lam2)

# Build the refractive index tensor (ellipsoid inverse square)
def build_index_tensor(n_o, n_e):
    return np.diag([1 / n_o**2, 1 / n_o**2, 1 / n_e**2])

# Rotation matrices
def Rx(theta):
    theta = theta * deg2rad
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def Ry(rho):
    rho = rho * deg2rad
    return np.array([[np.cos(rho), 0, np.sin(rho)],
                     [0, 1, 0],
                     [-np.sin(rho), 0, np.cos(rho)]])

def Rz(phi):
    phi = phi * deg2rad
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])

# Compute n_eff given a refractive index tensor and polarization
def n_eff(index_tensor, polarization):
    inv_n2 = polarization @ index_tensor @ polarization
    return 1 / np.sqrt(inv_n2)

# Δk calculation
def delta_k(theta=23.223, rho=0, phi=0):
    # Get rotated index tensors for both wavelengths
    R =  Rz(phi) @ Ry(rho) @ Rx(theta)  

    n_o_omega = n_o_BBO(lambda_fund)
    n_e_omega = n_e_BBO(lambda_fund)
    n_o_shg = n_o_BBO(lambda_shg)
    n_e_shg = n_e_BBO(lambda_shg)

    tensor_omega = R @ build_index_tensor(n_o_omega, n_e_omega) @ R.T
    tensor_shg = R @ build_index_tensor(n_o_shg, n_e_shg) @ R.T

    # Polarization vectors
    e_omega = np.array([1, 0, 0])   # x-polarized
    e_shg = np.array([0, 1, 0])     # y-polarized

    n_omega = n_eff(tensor_omega, e_omega)
    n_shg = n_eff(tensor_shg, e_shg)

    k_omega = 2 * np.pi / lambda_fund * n_omega
    k_shg = 2 * np.pi / lambda_shg * n_shg

    return 2 * k_omega - k_shg

def sinc_squared(delta_k):
    """Sinc squared function."""
    if delta_k == 0:
        return 1
    else:
        return ((np.sin(delta_k*1.5*1000/2) / (delta_k*1.5*1000/2))) ** 2

# Sweep over angle and plot Δk
"angles = np.linspace(0, 180, 1000)  # +/- 10 degrees"

"delta_k_theta = [delta_k(theta=a) for a in angles]"
"delta_k_rho = [delta_k(rho=a) for a in angles]"
"delta_k_phi = [delta_k(phi=a) for a in angles]"

""""""
# Plotting deta_k vs angles
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
angles = np.linspace(0,360, 1000)
delta_k_theta = [delta_k(theta=a) for a in angles]
plt.plot(angles, delta_k_theta)
plt.title("Δk vs θ (X-rotation)")
plt.xlabel("θ (deg)")
plt.ylabel("Δk (μm⁻¹)")
plt.grid(True)

plt.subplot(1, 3, 2)
angles = np.linspace(0, 360, 1000)
delta_k_rho = [delta_k(rho=a) for a in angles]
plt.plot(angles, delta_k_rho)
plt.title("Δk vs ρ (Y-rotation)")
plt.xlabel("ρ (deg)")
plt.grid(True)

plt.subplot(1, 3, 3)
angles = np.linspace(0, 360, 1000)
delta_k_phi = [delta_k(phi=a) for a in angles]
plt.plot(angles, delta_k_phi)
plt.title("Δk vs φ (Z-rotation)")
plt.xlabel("φ (deg)")
plt.grid(True)

plt.tight_layout()
plt.show()


# Plotting sinc²(Δk*1.5*1e3/2) vs angles
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
angles = np.linspace(17.3, 29.3, 1000)
delta_k_theta = [delta_k(theta=a) for a in angles]
plt.plot(angles, [sinc_squared(delta_k(a)) for a in angles])    
plt.title("Sinc²(Δk*L/2) vs θ (X-rotation)")
plt.xlabel("θ (deg)")
plt.ylabel("Sinc²(Δk*L/2)")
plt.grid(True)

plt.subplot(1, 3, 2)
angles = np.linspace(-6, 6, 1000)
delta_k_rho = [delta_k(rho=a) for a in angles]
plt.plot(angles, [sinc_squared(delta_k(rho=a)) for a in angles])
plt.title("Sinc²(Δk*L/2) vs ρ (Y-rotation)")
plt.xlabel("ρ (deg)")
plt.grid(True)

plt.subplot(1, 3, 3)
angles = np.linspace(-6,6, 1000)
delta_k_phi = [delta_k(phi=a) for a in angles]
plt.plot(angles, [sinc_squared(delta_k(phi=a)) for a in angles])
plt.title("Sinc²(Δk*L/2) vs φ (Z-rotation)")
plt.xlabel("φ (deg)")
plt.grid(True)
plt.tight_layout()
plt.show()

#different range of phi andgles
plt.subplot(1, 2, 1)
angles = np.linspace(-90, 270, 1000)
delta_k_phi = [delta_k(phi=a) for a in angles]
plt.plot(angles, [sinc_squared(delta_k(phi=a)) for a in angles])
plt.title("Sinc²(Δk*L/2) vs φ (Z-rotation)")
plt.xlabel("φ (deg)")
plt.grid(True)

plt.subplot(1, 2, 2)
angles = np.linspace(-5, 5, 1000)
delta_k_rho = [delta_k(phi=a) for a in angles]
plt.plot(angles, [sinc_squared(delta_k(phi=a)) for a in angles])
plt.title("Sinc²(Δk*L/2) vs φ (Z-rotation)")
plt.xlabel("φ (deg)")
plt.grid(True)
plt.tight_layout()
plt.show()