"""
Molecular Docking Simulator (OPTIMIZED)
========================================
Simulates drug-protein docking with realistic molecular representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleMolecule:
    """Simplified molecular representation for docking simulations."""
    
    def __init__(self, name, atoms):
        self.name = name
        self.atoms = atoms
        self.num_atoms = len(atoms)
        
    def get_center_of_mass(self):
        positions = np.array([atom['position'] for atom in self.atoms])
        return np.mean(positions, axis=0)


class ProteinBindingSite:
    """Simplified protein binding site representation."""
    
    def __init__(self, name, residues):
        self.name = name
        self.residues = residues
        self.num_residues = len(residues)
        
    def get_pocket_center(self):
        positions = np.array([res['position'] for res in self.residues])
        return np.mean(positions, axis=0)
    
    def get_pocket_size(self):
        positions = np.array([res['position'] for res in self.residues])
        distances = np.linalg.norm(positions - self.get_pocket_center(), axis=1)
        return np.max(distances)


class MolecularDockingSimulator:
    """Molecular docking simulator with quantum tunneling enhancement."""
    
    def __init__(self, protein, ligand, use_quantum=False):
        self.protein = protein
        self.ligand = ligand
        self.use_quantum = use_quantum
        
    def calculate_interaction_energy(self, ligand_position):
        """Calculate protein-ligand interaction energy."""
        energy = 0.0
        ligand_center = np.array(ligand_position)
        
        for residue in self.protein.residues:
            res_pos = np.array(residue['position'])
            distance = np.linalg.norm(ligand_center - res_pos)
            
            if distance < 0.1:
                distance = 0.1
            
            # Lennard-Jones potential (simplified)
            sigma = 0.35
            epsilon = 1.0
            vdw = 4 * epsilon * ((sigma/distance)**12 - (sigma/distance)**6)
            
            # Electrostatic (simplified)
            charge_ligand = -0.5
            charge_residue = 0.5 if residue['type'] == 'charged' else 0.0
            electrostatic = 138.935 * charge_ligand * charge_residue / distance
            
            # Hydrogen bonding (simplified)
            h_bond = 0.0
            if residue['type'] == 'polar' and distance < 0.35:
                h_bond = -20.0 * np.exp(-(distance - 0.25)**2 / 0.01)
            
            energy += vdw + electrostatic + h_bond
        
        # Desolvation penalty
        desolvation = 50.0 * np.exp(-np.linalg.norm(ligand_center - self.protein.get_pocket_center()) / 0.5)
        
        return energy + desolvation
    
    def perform_docking(self, num_poses=5, method='classical', show_progress=True):
        """Perform molecular docking to find best binding poses (OPTIMIZED)."""
        results = []
        pocket_center = self.protein.get_pocket_center()
        pocket_size = self.protein.get_pocket_size()
        
        iterator = tqdm(range(num_poses), desc=f"{method.upper()} docking") if show_progress else range(num_poses)
        
        for pose_idx in iterator:
            angle_offset = 2 * np.pi * pose_idx / num_poses
            radius_offset = pocket_size * 0.5
            
            start_pos = pocket_center + np.array([
                radius_offset * np.cos(angle_offset),
                radius_offset * np.sin(angle_offset),
                0.2 * (np.random.random() - 0.5)
            ])
            
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
        
        results.sort(key=lambda x: x['final_energy'])
        
        return {
            'method': method,
            'all_poses': results,
            'best_pose': results[0],
            'protein': self.protein.name,
            'ligand': self.ligand.name
        }
    
    def _optimize_classical(self, start_pos, max_iter=50):
        """Classical gradient descent optimization (OPTIMIZED)."""
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
    
    def _optimize_quantum(self, start_pos, max_iter=50):
        """Quantum-enhanced optimization with tunneling (OPTIMIZED)."""
        position = np.array(start_pos, dtype=float)
        trajectory = [position.copy()]
        learning_rate = 0.05
        stuck_counter = 0
        
        for iteration in range(max_iter):
            gradient = self._numerical_gradient(position)
            gradient_norm = np.linalg.norm(gradient)
            
            if gradient_norm < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            # Attempt tunneling if stuck
            if stuck_counter > 5:
                tunnel_prob = 0.3
                if np.random.random() < tunnel_prob:
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                    position = position + direction * 0.5
                    stuck_counter = 0
                    trajectory.append(position.copy())
                    continue
            
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
    """Create an example protein-ligand system for demonstration."""
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
    
    ligand_atoms = [
        {'element': 'C', 'position': [0, 0, 0], 'charge': 0, 'radius': 0.17},
        {'element': 'N', 'position': [0.15, 0.15, 0], 'charge': -0.3, 'radius': 0.155},
        {'element': 'O', 'position': [-0.15, 0.15, 0], 'charge': -0.4, 'radius': 0.152},
        {'element': 'C', 'position': [0.15, -0.15, 0], 'charge': 0, 'radius': 0.17},
    ]
    
    ligand = SimpleMolecule('Drug Candidate X', ligand_atoms)
    
    return protein, ligand


def visualize_docking_results(classical_results, quantum_results):
    """Visualize and compare docking results (OPTIMIZED)."""
    fig = plt.figure(figsize=(15, 10))
    
    protein_obj = create_example_system()[0]
    
    # 1. Best pose comparison - 3D
    ax1 = fig.add_subplot(231, projection='3d')
    
    for res in protein_obj.residues:
        pos = res['position']
        color = {'charged': 'red', 'polar': 'blue', 'hydrophobic': 'gray'}[res['type']]
        ax1.scatter(*pos, c=color, s=200, marker='o', alpha=0.6, edgecolors='black', linewidth=2)
    
    classical_best = classical_results['best_pose']['final_position']
    quantum_best = quantum_results['best_pose']['final_position']
    
    ax1.scatter(*classical_best, c='orange', s=300, marker='^', 
                edgecolors='black', linewidth=2, label='Classical', alpha=0.8)
    ax1.scatter(*quantum_best, c='lime', s=300, marker='*', 
                edgecolors='black', linewidth=2, label='Quantum', alpha=0.8)
    
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Z (nm)')
    ax1.set_title('Best Binding Poses', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. Energy distribution
    ax2 = fig.add_subplot(232)
    
    classical_energies = [pose['final_energy'] for pose in classical_results['all_poses']]
    quantum_energies = [pose['final_energy'] for pose in quantum_results['all_poses']]
    
    ax2.hist(classical_energies, bins=5, alpha=0.6, label='Classical', color='orange', edgecolor='black')
    ax2.hist(quantum_energies, bins=5, alpha=0.6, label='Quantum', color='lime', edgecolor='black')
    
    ax2.set_xlabel('Binding Energy (kJ/mol)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Energy Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Top poses comparison
    ax3 = fig.add_subplot(233)
    
    top_n = min(5, len(classical_results['all_poses']))
    classical_top = [pose['final_energy'] for pose in classical_results['all_poses'][:top_n]]
    quantum_top = [pose['final_energy'] for pose in quantum_results['all_poses'][:top_n]]
    
    x = np.arange(top_n)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, classical_top, width, label='Classical', 
                    color='orange', edgecolor='black')
    bars2 = ax3.bar(x + width/2, quantum_top, width, label='Quantum', 
                    color='lime', edgecolor='black')
    
    ax3.set_xlabel('Pose Rank')
    ax3.set_ylabel('Binding Energy (kJ/mol)')
    ax3.set_title('Top Poses Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'#{i+1}' for i in range(top_n)])
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Best energy comparison
    ax4 = fig.add_subplot(234)
    
    methods = ['Classical', 'Quantum']
    best_energies = [
        classical_results['best_pose']['final_energy'],
        quantum_results['best_pose']['final_energy']
    ]
    
    bars = ax4.bar(methods, best_energies, color=['orange', 'lime'], edgecolor='black', linewidth=2)
    
    for bar, energy in zip(bars, best_energies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    improvement = ((classical_results['best_pose']['final_energy'] - 
                   quantum_results['best_pose']['final_energy']) / 
                   abs(classical_results['best_pose']['final_energy']) * 100)
    
    ax4.set_ylabel('Best Binding Energy (kJ/mol)')
    ax4.set_title(f'Best Energy (Δ={improvement:.1f}%)', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # 5. Statistics summary
    ax5 = fig.add_subplot(235)
    ax5.axis('off')
    
    stats_text = f"""
DOCKING STATISTICS
{'='*30}

CLASSICAL:
  Best: {classical_results['best_pose']['final_energy']:.2f}
  Mean: {np.mean(classical_energies):.2f}
  
QUANTUM:
  Best: {quantum_results['best_pose']['final_energy']:.2f}
  Mean: {np.mean(quantum_energies):.2f}
  
IMPROVEMENT: {improvement:.1f}%

WINNER: {'🏆 QUANTUM' if improvement > 0 else '🏆 CLASSICAL'}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Summary plot
    ax6 = fig.add_subplot(236)
    
    data = [classical_energies, quantum_energies]
    bp = ax6.boxplot(data, labels=['Classical', 'Quantum'], patch_artist=True)
    bp['boxes'][0].set_facecolor('orange')
    bp['boxes'][1].set_facecolor('lime')
    
    ax6.set_ylabel('Binding Energy (kJ/mol)')
    ax6.set_title('Energy Distribution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('molecular_docking_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_complete_docking_study():
    """Run complete docking study comparing classical and quantum methods (OPTIMIZED)."""
    print("\n" + "="*70)
    print("MOLECULAR DOCKING STUDY")
    print("="*70)
    
    protein, ligand = create_example_system()
    
    print(f"\n✓ System: {protein.name} + {ligand.name}")
    
    # Classical docking
    print("\n🔬 Running CLASSICAL docking...")
    sim_classical = MolecularDockingSimulator(protein, ligand, use_quantum=False)
    classical_results = sim_classical.perform_docking(num_poses=5, method='classical')
    
    # Quantum docking
    print("\n⚛️  Running QUANTUM docking...")
    sim_quantum = MolecularDockingSimulator(protein, ligand, use_quantum=True)
    quantum_results = sim_quantum.perform_docking(num_poses=5, method='quantum')
    
    print("\n📊 Generating visualizations...")
    visualize_docking_results(classical_results, quantum_results)
    print("✓ Saved as 'molecular_docking_results.png'")
    
    return classical_results, quantum_results


if __name__ == "__main__":
    print("\n🧬 Starting Molecular Docking Simulations...\n")
    run_complete_docking_study()
    print("\n✓ Simulations completed!")