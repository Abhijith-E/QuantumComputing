"""
Quantum Tunneling Simulator - Core Module (OPTIMIZED)
======================================================
This module implements the fundamental quantum mechanics for tunneling calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.constants import hbar, m_e, e
from tqdm import tqdm

class QuantumTunnelingSimulator:
    """
    Simulates quantum tunneling through potential barriers using the WKB approximation.
    """
    
    def __init__(self, mass=1.67e-27):  # Default: proton mass in kg
        """
        Initialize the quantum tunneling simulator.
        
        Parameters:
        -----------
        mass : float
            Mass of the particle (default: proton mass in kg)
        """
        self.mass = mass
        self.hbar = hbar
        
    def calculate_tunneling_probability_rectangular(self, barrier_width, barrier_height, particle_energy):
        """
        Calculate tunneling probability through a rectangular barrier.
        Uses exact quantum mechanical formula.
        
        Parameters:
        -----------
        barrier_width : float
            Width of the barrier in meters
        barrier_height : float
            Height of the barrier in Joules
        particle_energy : float
            Energy of the particle in Joules
            
        Returns:
        --------
        float : Tunneling probability (0 to 1)
        """
        if particle_energy >= barrier_height:
            return 1.0  # Classical regime - particle goes over
        
        # Calculate wave vector inside barrier
        k = np.sqrt(2 * self.mass * (barrier_height - particle_energy)) / self.hbar
        
        # Tunneling probability for rectangular barrier
        T = np.exp(-2 * k * barrier_width)
        
        return T


class MolecularEnergyLandscape:
    """
    Creates realistic molecular energy landscapes for drug-protein interactions.
    """
    
    def __init__(self, num_dimensions=2):
        """
        Initialize molecular energy landscape.
        
        Parameters:
        -----------
        num_dimensions : int
            Number of spatial dimensions (2D or 3D)
        """
        self.num_dimensions = num_dimensions
        self.minima = []  # Local minima positions and energies
        self.barriers = []  # Energy barriers
        
    def add_binding_site(self, position, depth, width):
        """
        Add a binding site (energy minimum).
        
        Parameters:
        -----------
        position : array-like
            Position of the binding site
        depth : float
            Energy depth (negative for attractive)
        width : float
            Width of the binding well
        """
        self.minima.append({
            'position': np.array(position),
            'depth': depth,
            'width': width
        })
    
    def add_barrier(self, position, height, width):
        """
        Add an energy barrier.
        
        Parameters:
        -----------
        position : array-like
            Position of the barrier center
        height : float
            Barrier height (positive)
        width : float
            Width of the barrier
        """
        self.barriers.append({
            'position': np.array(position),
            'height': height,
            'width': width
        })
    
    def calculate_energy(self, position):
        """
        Calculate total energy at a given position.
        
        Parameters:
        -----------
        position : array-like
            Current position
            
        Returns:
        --------
        float : Total energy
        """
        position = np.array(position)
        energy = 0.0
        
        # Add contributions from binding sites (attractive)
        for minimum in self.minima:
            r = np.linalg.norm(position - minimum['position'])
            energy += minimum['depth'] * np.exp(-(r / minimum['width'])**2)
        
        # Add contributions from barriers (repulsive)
        for barrier in self.barriers:
            r = np.linalg.norm(position - barrier['position'])
            energy += barrier['height'] * np.exp(-(r / barrier['width'])**2)
        
        return energy
    
    def generate_2d_landscape(self, x_range, y_range, resolution=100):
        """
        Generate a 2D energy landscape for visualization.
        
        Parameters:
        -----------
        x_range : tuple
            (x_min, x_max)
        y_range : tuple
            (y_min, y_max)
        resolution : int
            Grid resolution
            
        Returns:
        --------
        X, Y, Z : ndarrays
            Meshgrid coordinates and energy values
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.calculate_energy([X[i, j], Y[i, j]])
        
        return X, Y, Z


def demo_tunneling_basics():
    """
    Demonstrate basic quantum tunneling calculations.
    """
    print("=" * 70)
    print("QUANTUM TUNNELING DEMONSTRATION")
    print("=" * 70)
    
    # Initialize simulator
    simulator = QuantumTunnelingSimulator(mass=1.67e-27)  # Proton mass
    
    # Define barrier parameters (in atomic units for simplicity)
    barrier_width = 1e-10  # 1 Angstrom in meters
    barrier_height = 1e-19  # ~0.6 eV in Joules
    
    print("\nBarrier Parameters:")
    print(f"  Width: {barrier_width*1e10:.2f} Angstroms")
    print(f"  Height: {barrier_height/e:.3f} eV")
    
    # Calculate tunneling for different energies
    energies = np.linspace(0.1, 0.9, 5) * barrier_height  # Reduced from 9 to 5
    probabilities = []
    
    print("\nTunneling Probabilities:")
    print("-" * 50)
    print(f"{'Energy (eV)':<15} {'Probability':<15} {'Percentage'}")
    print("-" * 50)
    
    for energy in energies:
        prob = simulator.calculate_tunneling_probability_rectangular(
            barrier_width, barrier_height, energy
        )
        probabilities.append(prob)
        print(f"{energy/e:>10.3f}     {prob:>10.6e}     {prob*100:>8.4f}%")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Tunneling probability vs energy
    ax1.plot(energies/e, probabilities, 'b-', linewidth=2, marker='o')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='50% probability')
    ax1.set_xlabel('Particle Energy (eV)', fontsize=12)
    ax1.set_ylabel('Tunneling Probability', fontsize=12)
    ax1.set_title('Quantum Tunneling Probability vs Energy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Potential barrier visualization (simplified)
    x_sim = np.linspace(-2*barrier_width, 3*barrier_width, 200)  # Reduced from 1000
    V_sim = np.zeros_like(x_sim)
    for i, xi in enumerate(x_sim):
        if 0 <= xi <= barrier_width:
            V_sim[i] = barrier_height
    
    particle_energy = 0.5 * barrier_height
    
    ax2.plot(x_sim*1e10, V_sim/e, 'r-', linewidth=2, label='Potential Barrier')
    ax2.axhline(y=particle_energy/e, color='g', linestyle='--', label='Particle Energy', linewidth=2)
    ax2.fill_between(x_sim*1e10, 0, V_sim/e, alpha=0.3, color='red')
    ax2.set_xlabel('Position (Angstroms)', fontsize=12)
    ax2.set_ylabel('Energy (eV)', fontsize=12)
    ax2.set_title('Potential Barrier Structure', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, barrier_height/e * 1.2)
    
    plt.tight_layout()
    plt.savefig('quantum_tunneling_basics.png', dpi=150, bbox_inches='tight')  # Reduced DPI
    print("\n✓ Visualization saved as 'quantum_tunneling_basics.png'")
    plt.close()


def demo_molecular_landscape():
    """
    Demonstrate molecular energy landscape creation.
    """
    print("\n" + "=" * 70)
    print("MOLECULAR ENERGY LANDSCAPE DEMONSTRATION")
    print("=" * 70)
    
    # Create landscape
    landscape = MolecularEnergyLandscape(num_dimensions=2)
    
    # Add binding sites (local minima)
    landscape.add_binding_site(position=[2, 2], depth=-5.0, width=0.8)  # Deep site
    landscape.add_binding_site(position=[6, 5], depth=-8.0, width=1.0)  # Deeper site (global minimum)
    landscape.add_binding_site(position=[4, 7], depth=-4.0, width=0.6)  # Shallow site
    
    # Add barriers
    landscape.add_barrier(position=[4, 3.5], height=6.0, width=0.8)  # Main barrier
    landscape.add_barrier(position=[5, 6], height=3.0, width=0.5)    # Smaller barrier
    
    print("\nLandscape Configuration:")
    print(f"  Number of binding sites: {len(landscape.minima)}")
    print(f"  Number of barriers: {len(landscape.barriers)}")
    
    # Generate landscape (reduced resolution)
    X, Y, Z = landscape.generate_2d_landscape(x_range=(0, 8), y_range=(0, 8), resolution=80)
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('X Position', fontsize=10)
    ax1.set_ylabel('Y Position', fontsize=10)
    ax1.set_zlabel('Energy', fontsize=10)
    ax1.set_title('3D Energy Landscape', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='Energy')
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    # Mark minima and barriers
    for site in landscape.minima:
        ax2.plot(site['position'][0], site['position'][1], 'r*', markersize=15)
    for barrier in landscape.barriers:
        ax2.plot(barrier['position'][0], barrier['position'][1], 'wx', markersize=12, markeredgewidth=2)
    
    ax2.set_xlabel('X Position', fontsize=10)
    ax2.set_ylabel('Y Position', fontsize=10)
    ax2.set_title('2D Contour Map', fontsize=12, fontweight='bold')
    fig.colorbar(contour, ax=ax2, label='Energy')
    
    # Energy profile along a line
    ax3 = fig.add_subplot(133)
    line_x = np.linspace(0, 8, 100)  # Reduced from 200
    line_y = 4.0
    line_energies = [landscape.calculate_energy([x, line_y]) for x in line_x]
    
    ax3.plot(line_x, line_energies, 'b-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.fill_between(line_x, line_energies, 0, where=(np.array(line_energies) > 0), alpha=0.3, color='red', label='Barrier')
    ax3.fill_between(line_x, line_energies, 0, where=(np.array(line_energies) < 0), alpha=0.3, color='green', label='Well')
    ax3.set_xlabel('X Position (Y=4.0)', fontsize=10)
    ax3.set_ylabel('Energy', fontsize=10)
    ax3.set_title('Energy Profile Along Path', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('molecular_energy_landscape.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'molecular_energy_landscape.png'")
    plt.close()


if __name__ == "__main__":
    print("\n🚀 Starting Quantum Tunneling Simulations...\n")
    
    # Run demonstrations
    demo_tunneling_basics()
    demo_molecular_landscape()
    
    print("\n" + "=" * 70)
    print("✓ All demonstrations completed successfully!")
    print("=" * 70)