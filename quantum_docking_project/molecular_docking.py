"""
Molecular Docking Simulator
============================
Simulates drug-protein docking with realistic molecular representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import json


class SimpleMolecule:
    """
    Simplified molecular representation for docking simulations.
    """
    
    def __init__(self, name, atoms):
        """
        Initialize molecule.
        
        Parameters:
        -----------
        name : str
            Molecule name
        atoms : list of dict
            List of atoms with 'element', 'position', 'charge', 'radius'
        """
        self.name = name
        self.atoms = atoms
        self.num_atoms = len(atoms)
        
    def get_center_of_mass(self):
        """Calculate center of mass."""
        positions = np.array([atom['position'] for atom in self.atoms])
        return np.mean(positions, axis=0)
    
    def translate(self, vector):
        """Translate molecule by vector."""
        for atom in self.atoms:
            atom['position'] = np.array(atom['position']) + np.array(vector)
    
    def rotate(self, angle, axis='z'):
        """Rotate molecule around axis."""
        center = self.get_center_of_mass()
        
        # Rotation matrices
        if axis == 'z':
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        elif axis == 'y':
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        else:  # x
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        
        for atom in self.atoms:
            pos = np.array(atom['position']) - center
            rotated = R @ pos
            atom['position'] = rotated + center
    
    def get_bounding_box(self):
        """Get bounding box dimensions."""
        positions = np.array([atom['position'] for atom in self.atoms])
        mins = np.min(positions, axis=0)
        maxs = np.max(positions, axis=0)
        return mins, maxs


class ProteinBindingSite:
    """
    Simplified protein binding site representation.
    """
    
    def __init__(self, name, residues):
        """
        Initialize protein binding site.
        
        Parameters:
        -----------
        name : str
            Protein name
        residues : list of dict
            Residues with 'name', 'position', 'type' (hydrophobic/polar/charged)
        """
        self.name = name
        self.residues = residues
        self.num_residues = len(residues)
        
    def get_pocket_center(self):
        """Get center of binding pocket."""
        positions = np.array([res['position'] for res in self.residues])
        return np.mean(positions, axis=0)
    
    def get_pocket_size(self):
        """Estimate pocket size."""
        positions = np.array([res['position'] for res in self.residues])
        distances = np.linalg.norm(positions - self.get_pocket_center(), axis=1)
        return np.max(distances)


class MolecularDockingSimulator:
    """
    Molecular docking simulator with quantum tunneling enhancement.
    """
    
    def __init__(self, protein, ligand, use_quantum=False):
        """
        Initialize docking simulator.
        
        Parameters:
        -----------
        protein : ProteinBindingSite
            Protein binding site
        ligand : SimpleMolecule
            Ligand molecule to dock
        use_quantum : bool
            Whether to use quantum tunneling
        """
        self.protein = protein
        self.ligand = ligand
        self.use_quantum = use_quantum
        self.docking_history = []
        
    def calculate_interaction_energy(self, ligand_position):
        """
        Calculate protein-ligand interaction energy.
        
        Parameters:
        -----------
        ligand_position : array-like
            Center position of ligand
            
        Returns:
        --------
        float : Interaction energy
        """
        energy = 0.0
        ligand_center = np.array(ligand_position)
        
        # Van der Waals and electrostatic interactions
        for residue in self.protein.residues:
            res_pos = np.array(residue['position'])
            distance = np.linalg.norm(ligand_center - res_pos)
            
            if distance < 0.1:  # Avoid singularity
                distance = 0.1
            
            # Lennard-Jones potential (simplified)
            sigma = 0.35  # nm
            epsilon = 1.0  # kJ/mol
            
            vdw = 4 * epsilon * ((sigma/distance)**12 - (sigma/distance)**6)
            
            # Electrostatic (simplified)
            charge_ligand = -0.5  # Simplified
            charge_residue = 0.5 if residue['type'] == 'charged' else 0.0
            electrostatic = 138.935 * charge_ligand * charge_residue / distance  # kJ/mol
            
            # Hydrogen bonding (simplified)
            h_bond = 0.0
            if residue['type'] == 'polar' and distance < 0.35:
                h_bond = -20.0 * np.exp(-(distance - 0.25)**2 / 0.01)
            
            energy += vdw + electrostatic + h_bond
        
        # Desolvation penalty (simplified)
        desolvation = 50.0 * np.exp(-np.linalg.norm(ligand_center - self.protein.get_pocket_center()) / 0.5)
        
        return energy + desolvation
    
    def create_energy_landscape_2d(self, x_range, y_range, z_fixed=0, resolution=50):
        """
        Create 2D energy landscape for visualization.
        
        Parameters:
        -----------
        x_range, y_range : tuple
            (min, max) for x and y coordinates
        z_fixed : float
            Fixed z coordinate
        resolution : int
            Grid resolution
            
        Returns:
        --------
        X, Y, Z : ndarrays
            Meshgrid and energy values
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.calculate_interaction_energy([X[i, j], Y[i, j], z_fixed])
        
        return X, Y, Z
    
    def perform_docking(self, num_poses=10, method='quantum' if None else 'classical'):
        """
        Perform molecular docking to find best binding poses.
        
        Parameters:
        -----------
        num_poses : int
            Number of different starting poses to try
        method : str
            'classical' or 'quantum'
            
        Returns:
        --------
        dict : Docking results
        """
        print(f"\n{'='*70}")
        print(f"MOLECULAR DOCKING SIMULATION - {method.upper()} METHOD")
        print(f"{'='*70}")
        print(f"Protein: {self.protein.name}")
        print(f"Ligand: {self.ligand.name}")
        print(f"Number of poses: {num_poses}")
        
        results = []
        pocket_center = self.protein.get_pocket_center()
        pocket_size = self.protein.get_pocket_size()
        
        for pose_idx in range(num_poses):
            # Generate random starting position near pocket
            angle_offset = 2 * np.pi * pose_idx / num_poses
            radius_offset = pocket_size * 0.5
            
            start_pos = pocket_center + np.array([
                radius_offset * np.cos(angle_offset),
                radius_offset * np.sin(angle_offset),
                0.2 * (np.random.random() - 0.5)
            ])
            
            # Random rotation
            rotation_angle = np.random.uniform(0, 2*np.pi)
            
            print(f"\nPose {pose_idx + 1}/{num_poses}:")
            print(f"  Start position: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
            
            # Optimize pose
            if method == 'quantum':
                final_pos, final_energy, trajectory = self._optimize_quantum(start_pos)
            else:
                final_pos, final_energy, trajectory = self._optimize_classical(start_pos)
            
            results.append({
                'pose_id': pose_idx,
                'start_position': start_pos,
                'final_position': final_pos,
                'final_energy': final_energy,
                'trajectory': trajectory
            })
            
            print(f"  Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
            print(f"  Final energy: {final_energy:.2f} kJ/mol")
        
        # Sort by energy
        results.sort(key=lambda x: x['final_energy'])
        
        print(f"\n{'='*70}")
        print("TOP 3 BINDING POSES:")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'Pose ID':<10} {'Energy (kJ/mol)':<20} {'Position'}")
        print(f"{'-'*70}")
        
        for rank, result in enumerate(results[:3], 1):
            pos = result['final_position']
            print(f"{rank:<6} {result['pose_id']:<10} {result['final_energy']:>15.2f}     "
                  f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        
        return {
            'method': method,
            'all_poses': results,
            'best_pose': results[0],
            'protein': self.protein.name,
            'ligand': self.ligand.name
        }
    
    def _optimize_classical(self, start_pos, max_iter=100):
        """Classical gradient descent optimization."""
        position = np.array(start_pos, dtype=float)
        trajectory = [position.copy()]
        learning_rate = 0.05
        
        for _ in range(max_iter):
            gradient = self._numerical_gradient(position)
            position = position - learning_rate * gradient
            trajectory.append(position.copy())
            
            if np.linalg.norm(gradient) < 1e-3:
                break
        
        final_energy = self.calculate_interaction_energy(position)
        return position, final_energy, np.array(trajectory)
    
    def _optimize_quantum(self, start_pos, max_iter=100):
        """Quantum-enhanced optimization with tunneling."""
        position = np.array(start_pos, dtype=float)
        trajectory = [position.copy()]
        learning_rate = 0.05
        stuck_counter = 0
        
        for iteration in range(max_iter):
            gradient = self._numerical_gradient(position)
            gradient_norm = np.linalg.norm(gradient)
            
            # Check if stuck
            if gradient_norm < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            # Attempt tunneling if stuck
            if stuck_counter > 5:
                tunnel_prob = 0.3  # Simplified tunneling probability
                if np.random.random() < tunnel_prob:
                    # Tunnel to random nearby position
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                    position = position + direction * 0.5
                    stuck_counter = 0
                    trajectory.append(position.copy())
                    continue
            
            # Normal gradient step
            position = position - learning_rate * gradient
            trajectory.append(position.copy())
            
            if gradient_norm < 1e-3:
                break
        
        final_energy = self.calculate_interaction_energy(position)
        return position, final_energy, np.array(trajectory)
    
    def _numerical_gradient(self, position, h=1e-4):
        """Calculate numerical gradient of energy."""
        gradient = np.zeros(3)
        
        for i in range(3):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += h
            pos_minus[i] -= h
            
            energy_plus = self.calculate_interaction_energy(pos_plus)
            energy_minus = self.calculate_interaction_energy(pos_minus)
            
            gradient[i] = (energy_plus - energy_minus) / (2 * h)
        
        return gradient


def create_example_system():
    """
    Create an example protein-ligand system for demonstration.
    """
    # Create protein binding site (simplified enzyme active site)
    protein_residues = [
        {'name': 'ASP189', 'position': [0.5, 0.5, 0], 'type': 'charged'},
        {'name': 'HIS57', 'position': [0.3, -0.4, 0.1], 'type': 'polar'},
        {'name': 'SER195', 'position': [-0.4, 0.3, -0.1], 'type': 'polar'},
        {'name': 'GLY193', 'position': [0.6, -0.2, 0.2], 'type': 'hydrophobic'},
        {'name': 'ALA190', 'position': [-0.3, -0.5, 0], 'type': 'hydrophobic'},
        {'name': 'VAL213', 'position': [0, 0.7, -0.2], 'type': 'hydrophobic'},
        {'name': 'PHE41', 'position': [-0.6, 0, 0.1], 'type': 'hydrophobic'},
    ]
    
    protein = ProteinBindingSite('Example Enzyme', protein_residues)
    
    # Create ligand (simplified drug molecule)
    ligand_atoms = [
        {'element': 'C', 'position': [0, 0, 0], 'charge': 0, 'radius': 0.17},
        {'element': 'N', 'position': [0.15, 0.15, 0], 'charge': -0.3, 'radius': 0.155},
        {'element': 'O', 'position': [-0.15, 0.15, 0], 'charge': -0.4, 'radius': 0.152},
        {'element': 'C', 'position': [0.15, -0.15, 0], 'charge': 0, 'radius': 0.17},
    ]
    
    ligand = SimpleMolecule('Drug Candidate X', ligand_atoms)
    
    return protein, ligand


def visualize_docking_results(classical_results, quantum_results):
    """
    Visualize and compare docking results.
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Get protein and ligand
    protein = classical_results['all_poses'][0]  # Just for structure
    
    # 1. Best pose comparison - 3D
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Plot protein residues
    protein_obj = create_example_system()[0]
    for res in protein_obj.residues:
        pos = res['position']
        color = {'charged': 'red', 'polar': 'blue', 'hydrophobic': 'gray'}[res['type']]
        ax1.scatter(*pos, c=color, s=200, marker='o', edgecolors='black', linewidth=2, alpha=0.6)
    
    # Plot best ligand positions
    classical_best = classical_results['best_pose']['final_position']
    quantum_best = quantum_results['best_pose']['final_position']
    
    ax1.scatter(*classical_best, c='orange', s=300, marker='^', 
                edgecolors='black', linewidth=2, label='Classical Best', alpha=0.8)
    ax1.scatter(*quantum_best, c='lime', s=300, marker='*', 
                edgecolors='black', linewidth=2, label='Quantum Best', alpha=0.8)
    
    ax1.set_xlabel('X (nm)', fontsize=10)
    ax1.set_ylabel('Y (nm)', fontsize=10)
    ax1.set_zlabel('Z (nm)', fontsize=10)
    ax1.set_title('Best Binding Poses (3D)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    
    # 2. Energy distribution
    ax2 = fig.add_subplot(232)
    
    classical_energies = [pose['final_energy'] for pose in classical_results['all_poses']]
    quantum_energies = [pose['final_energy'] for pose in quantum_results['all_poses']]
    
    ax2.hist(classical_energies, bins=10, alpha=0.6, label='Classical', color='orange', edgecolor='black')
    ax2.hist(quantum_energies, bins=10, alpha=0.6, label='Quantum', color='lime', edgecolor='black')
    ax2.axvline(np.mean(classical_energies), color='orange', linestyle='--', linewidth=2, label='Classical Mean')
    ax2.axvline(np.mean(quantum_energies), color='lime', linestyle='--', linewidth=2, label='Quantum Mean')
    
    ax2.set_xlabel('Binding Energy (kJ/mol)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Energy Distribution of All Poses', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Top 5 poses comparison
    ax3 = fig.add_subplot(233)
    
    top_n = 5
    classical_top = [pose['final_energy'] for pose in classical_results['all_poses'][:top_n]]
    quantum_top = [pose['final_energy'] for pose in quantum_results['all_poses'][:top_n]]
    
    x = np.arange(top_n)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, classical_top, width, label='Classical', 
                    color='orange', edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, quantum_top, width, label='Quantum', 
                    color='lime', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_xlabel('Pose Rank', fontsize=11)
    ax3.set_ylabel('Binding Energy (kJ/mol)', fontsize=11)
    ax3.set_title('Top 5 Poses Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'#{i+1}' for i in range(top_n)])
    ax3.legend(fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Best energy comparison
    ax4 = fig.add_subplot(234)
    
    methods = ['Classical', 'Quantum']
    best_energies = [
        classical_results['best_pose']['final_energy'],
        quantum_results['best_pose']['final_energy']
    ]
    
    colors_bar = ['orange', 'lime']
    bars = ax4.bar(methods, best_energies, color=colors_bar, edgecolor='black', linewidth=2, width=0.6)
    
    for bar, energy in zip(bars, best_energies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.2f}\nkJ/mol', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    improvement = ((classical_results['best_pose']['final_energy'] - 
                   quantum_results['best_pose']['final_energy']) / 
                   abs(classical_results['best_pose']['final_energy']) * 100)
    
    ax4.set_ylabel('Best Binding Energy (kJ/mol)', fontsize=11)
    ax4.set_title(f'Best Pose Energy\n(Quantum improvement: {improvement:.1f}%)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # 5. Statistical summary
    ax5 = fig.add_subplot(235)
    ax5.axis('off')
    
    stats_text = f"""
    DOCKING STATISTICS SUMMARY
    {'='*40}
    
    CLASSICAL METHOD:
    • Best Energy: {classical_results['best_pose']['final_energy']:.2f} kJ/mol
    • Mean Energy: {np.mean(classical_energies):.2f} kJ/mol
    • Std Dev: {np.std(classical_energies):.2f} kJ/mol
    • Worst Energy: {max(classical_energies):.2f} kJ/mol
    
    QUANTUM METHOD:
    • Best Energy: {quantum_results['best_pose']['final_energy']:.2f} kJ/mol
    • Mean Energy: {np.mean(quantum_energies):.2f} kJ/mol
    • Std Dev: {np.std(quantum_energies):.2f} kJ/mol
    • Worst Energy: {max(quantum_energies):.2f} kJ/mol
    
    IMPROVEMENT:
    • Energy Improvement: {improvement:.2f}%
    • Mean Improvement: {((np.mean(classical_energies) - np.mean(quantum_energies))/abs(np.mean(classical_energies))*100):.2f}%
    
    WINNER: {'🏆 QUANTUM' if quantum_results['best_pose']['final_energy'] < classical_results['best_pose']['final_energy'] else '🏆 CLASSICAL'}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Convergence comparison (best trajectory)
    ax6 = fig.add_subplot(236)
    
    # Get energies along trajectory for best poses
    def get_trajectory_energies(trajectory, simulator):
        return [simulator.calculate_interaction_energy(pos) for pos in trajectory]
    
    # Recreate simulator to calculate energies
    protein, ligand = create_example_system()
    sim_classical = MolecularDockingSimulator(protein, ligand, use_quantum=False)
    sim_quantum = MolecularDockingSimulator(protein, ligand, use_quantum=True)
    
    classical_traj = classical_results['best_pose']['trajectory']
    quantum_traj = quantum_results['best_pose']['trajectory']
    
    classical_traj_energies = get_trajectory_energies(classical_traj, sim_classical)
    quantum_traj_energies = get_trajectory_energies(quantum_traj, sim_quantum)
    
    ax6.plot(classical_traj_energies, color='orange', linewidth=2, label='Classical', marker='o', markersize=4)
    ax6.plot(quantum_traj_energies, color='lime', linewidth=2, label='Quantum', marker='s', markersize=4)
    
    ax6.set_xlabel('Optimization Step', fontsize=11)
    ax6.set_ylabel('Energy (kJ/mol)', fontsize=11)
    ax6.set_title('Best Pose Convergence', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('molecular_docking_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'molecular_docking_results.png'")
    plt.show()


def run_complete_docking_study():
    """
    Run complete docking study comparing classical and quantum methods.
    """
    print("\n" + "="*70)
    print("COMPLETE MOLECULAR DOCKING STUDY")
    print("="*70)
    
    # Create system
    print("\n📦 Creating molecular system...")
    protein, ligand = create_example_system()
    
    print(f"✓ Protein: {protein.name}")
    print(f"  - Binding site residues: {protein.num_residues}")
    print(f"  - Pocket center: {protein.get_pocket_center()}")
    print(f"  - Pocket size: {protein.get_pocket_size():.3f} nm")
    
    print(f"\n✓ Ligand: {ligand.name}")
    print(f"  - Number of atoms: {ligand.num_atoms}")
    
    # Classical docking
    print("\n🔬 Performing CLASSICAL docking...")
    sim_classical = MolecularDockingSimulator(protein, ligand, use_quantum=False)
    classical_results = sim_classical.perform_docking(num_poses=10, method='classical')
    
    # Quantum docking
    print("\n⚛️  Performing QUANTUM-ENHANCED docking...")
    sim_quantum = MolecularDockingSimulator(protein, ligand, use_quantum=True)
    quantum_results = sim_quantum.perform_docking(num_poses=10, method='quantum')
    
    # Visualize comparison
    print("\n📊 Generating comparison visualizations...")
    visualize_docking_results(classical_results, quantum_results)
    
    print("\n✅ Complete docking study finished!")
    
    return classical_results, quantum_results


if __name__ == "__main__":
    print("\n🧬 Starting Molecular Docking Simulations...\n")
    classical_results, quantum_results = run_complete_docking_study()
    print("\n" + "="*70)
    print("✓ All simulations completed successfully!")
    print("="*70)