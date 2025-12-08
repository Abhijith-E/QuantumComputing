"""
Optimization Algorithms Module
================================
Implements classical and quantum-enhanced optimization algorithms for molecular docking.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time


class ClassicalOptimizer:
    """
    Classical optimization using gradient descent and simulated annealing.
    """
    
    def __init__(self, energy_landscape):
        """
        Initialize classical optimizer.
        
        Parameters:
        -----------
        energy_landscape : MolecularEnergyLandscape
            The energy landscape to optimize over
        """
        self.landscape = energy_landscape
        self.history = []
        
    def gradient_descent(self, start_position, learning_rate=0.1, max_iterations=500, tolerance=1e-6):
        """
        Perform gradient descent optimization.
        
        Parameters:
        -----------
        start_position : array-like
            Starting position
        learning_rate : float
            Step size for gradient descent
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        dict : Optimization results including trajectory and final position
        """
        position = np.array(start_position, dtype=float)
        self.history = [position.copy()]
        energies = [self.landscape.calculate_energy(position)]
        
        for iteration in range(max_iterations):
            # Numerical gradient
            gradient = self._numerical_gradient(position)
            
            # Update position
            new_position = position - learning_rate * gradient
            new_energy = self.landscape.calculate_energy(new_position)
            
            # Store history
            self.history.append(new_position.copy())
            energies.append(new_energy)
            
            # Check convergence
            if np.linalg.norm(gradient) < tolerance:
                break
            
            position = new_position
        
        return {
            'final_position': position,
            'final_energy': energies[-1],
            'trajectory': np.array(self.history),
            'energies': np.array(energies),
            'iterations': len(self.history),
            'converged': iteration < max_iterations - 1
        }
    
    def simulated_annealing(self, start_position, initial_temp=100.0, cooling_rate=0.95, 
                           max_iterations=1000, step_size=0.5):
        """
        Perform simulated annealing optimization.
        
        Parameters:
        -----------
        start_position : array-like
            Starting position
        initial_temp : float
            Initial temperature
        cooling_rate : float
            Temperature cooling rate
        max_iterations : int
            Maximum number of iterations
        step_size : float
            Maximum step size for random moves
            
        Returns:
        --------
        dict : Optimization results
        """
        position = np.array(start_position, dtype=float)
        energy = self.landscape.calculate_energy(position)
        
        best_position = position.copy()
        best_energy = energy
        
        self.history = [position.copy()]
        energies = [energy]
        temperature = initial_temp
        
        accepted_moves = 0
        
        for iteration in range(max_iterations):
            # Random neighboring position
            delta = np.random.uniform(-step_size, step_size, size=position.shape)
            new_position = position + delta
            new_energy = self.landscape.calculate_energy(new_position)
            
            # Acceptance criterion
            delta_energy = new_energy - energy
            
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                position = new_position
                energy = new_energy
                accepted_moves += 1
                
                # Update best
                if energy < best_energy:
                    best_position = position.copy()
                    best_energy = energy
            
            self.history.append(position.copy())
            energies.append(energy)
            
            # Cool down
            temperature *= cooling_rate
        
        return {
            'final_position': best_position,
            'final_energy': best_energy,
            'trajectory': np.array(self.history),
            'energies': np.array(energies),
            'iterations': len(self.history),
            'acceptance_rate': accepted_moves / max_iterations
        }
    
    def _numerical_gradient(self, position, h=1e-5):
        """
        Calculate numerical gradient using finite differences.
        """
        gradient = np.zeros_like(position)
        
        for i in range(len(position)):
            position_plus = position.copy()
            position_minus = position.copy()
            
            position_plus[i] += h
            position_minus[i] -= h
            
            energy_plus = self.landscape.calculate_energy(position_plus)
            energy_minus = self.landscape.calculate_energy(position_minus)
            
            gradient[i] = (energy_plus - energy_minus) / (2 * h)
        
        return gradient


class QuantumEnhancedOptimizer:
    """
    Quantum tunneling-enhanced optimization algorithm.
    """
    
    def __init__(self, energy_landscape, tunneling_simulator):
        """
        Initialize quantum-enhanced optimizer.
        
        Parameters:
        -----------
        energy_landscape : MolecularEnergyLandscape
            The energy landscape to optimize over
        tunneling_simulator : QuantumTunnelingSimulator
            Quantum tunneling calculator
        """
        self.landscape = energy_landscape
        self.tunneling_sim = tunneling_simulator
        self.history = []
        self.tunnel_events = []
        
    def quantum_gradient_descent(self, start_position, learning_rate=0.1, max_iterations=500,
                                tolerance=1e-6, tunnel_threshold=2.0, tunnel_distance=1.0):
        """
        Gradient descent with quantum tunneling capability.
        
        Parameters:
        -----------
        start_position : array-like
            Starting position
        learning_rate : float
            Step size for gradient descent
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        tunnel_threshold : float
            Energy threshold to attempt tunneling
        tunnel_distance : float
            Distance to tunnel when stuck
            
        Returns:
        --------
        dict : Optimization results
        """
        position = np.array(start_position, dtype=float)
        self.history = [position.copy()]
        energies = [self.landscape.calculate_energy(position)]
        self.tunnel_events = []
        
        stuck_counter = 0
        
        for iteration in range(max_iterations):
            # Calculate gradient
            gradient = self._numerical_gradient(position)
            gradient_norm = np.linalg.norm(gradient)
            
            # Try normal gradient descent
            new_position = position - learning_rate * gradient
            new_energy = self.landscape.calculate_energy(new_position)
            current_energy = energies[-1]
            
            # Check if stuck at local minimum
            if gradient_norm < tolerance * 10:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            # Attempt quantum tunneling if stuck
            if stuck_counter > 5 and current_energy > -tunnel_threshold:
                tunnel_attempted = True
                tunneled_position, tunneled, tunnel_prob = self._attempt_tunneling(
                    position, gradient, tunnel_distance
                )
                
                if tunneled:
                    position = tunneled_position
                    new_energy = self.landscape.calculate_energy(position)
                    self.tunnel_events.append({
                        'iteration': iteration,
                        'from': self.history[-1],
                        'to': position,
                        'probability': tunnel_prob
                    })
                    stuck_counter = 0
                    print(f"  🌀 Tunneling event at iteration {iteration}! (p={tunnel_prob:.4f})")
                else:
                    position = new_position
            else:
                position = new_position
            
            self.history.append(position.copy())
            energies.append(new_energy)
            
            # Check convergence
            if gradient_norm < tolerance and stuck_counter == 0:
                break
        
        return {
            'final_position': position,
            'final_energy': energies[-1],
            'trajectory': np.array(self.history),
            'energies': np.array(energies),
            'iterations': len(self.history),
            'tunnel_events': self.tunnel_events,
            'num_tunneling': len(self.tunnel_events)
        }
    
    def _attempt_tunneling(self, position, gradient, tunnel_distance):
        """
        Attempt to tunnel through a nearby barrier.
        
        Returns:
        --------
        tuple : (new_position, tunneled_success, probability)
        """
        current_energy = self.landscape.calculate_energy(position)
        
        # Try tunneling in the direction opposite to gradient (toward lower energy beyond barrier)
        if np.linalg.norm(gradient) > 1e-10:
            direction = -gradient / np.linalg.norm(gradient)
        else:
            # Random direction if gradient is zero
            direction = np.random.randn(len(position))
            direction = direction / np.linalg.norm(direction)
        
        # Sample points along the tunneling direction
        test_position = position + direction * tunnel_distance
        barrier_energy = current_energy
        
        # Find maximum energy along the path (barrier height)
        n_samples = 20
        for alpha in np.linspace(0, 1, n_samples):
            test_pos = position + alpha * direction * tunnel_distance
            test_energy = self.landscape.calculate_energy(test_pos)
            if test_energy > barrier_energy:
                barrier_energy = test_energy
        
        barrier_height = (barrier_energy - current_energy) * 1.6e-19  # Convert to Joules (rough scale)
        barrier_width = tunnel_distance * 1e-10  # Convert to meters (rough scale)
        particle_energy = current_energy * 1.6e-19 * 0.8  # Particle has less energy than barrier
        
        # Calculate tunneling probability
        if barrier_height > particle_energy:
            tunnel_prob = self.tunneling_sim.calculate_tunneling_probability_rectangular(
                barrier_width, abs(barrier_height), abs(particle_energy)
            )
        else:
            tunnel_prob = 1.0
        
        # Decide whether tunneling succeeds
        if np.random.random() < tunnel_prob:
            return test_position, True, tunnel_prob
        else:
            return position, False, tunnel_prob
    
    def _numerical_gradient(self, position, h=1e-5):
        """Calculate numerical gradient."""
        gradient = np.zeros_like(position)
        
        for i in range(len(position)):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += h
            pos_minus[i] -= h
            
            gradient[i] = (self.landscape.calculate_energy(pos_plus) - 
                          self.landscape.calculate_energy(pos_minus)) / (2 * h)
        
        return gradient


def compare_optimizers(landscape, start_position):
    """
    Compare classical and quantum-enhanced optimizers.
    
    Parameters:
    -----------
    landscape : MolecularEnergyLandscape
        Energy landscape to optimize
    start_position : array-like
        Starting position for optimization
        
    Returns:
    --------
    dict : Results from all optimization methods
    """
    from quantum_tunneling_core import QuantumTunnelingSimulator
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPARISON")
    print("=" * 70)
    print(f"\nStarting position: {start_position}")
    print(f"Starting energy: {landscape.calculate_energy(start_position):.4f}")
    
    results = {}
    
    # Classical gradient descent
    print("\n1. Running Classical Gradient Descent...")
    classical_opt = ClassicalOptimizer(landscape)
    start_time = time.time()
    results['gradient_descent'] = classical_opt.gradient_descent(start_position, learning_rate=0.15)
    results['gradient_descent']['time'] = time.time() - start_time
    print(f"   ✓ Completed in {results['gradient_descent']['time']:.3f}s")
    print(f"   Final energy: {results['gradient_descent']['final_energy']:.4f}")
    print(f"   Iterations: {results['gradient_descent']['iterations']}")
    
    # Simulated annealing
    print("\n2. Running Simulated Annealing...")
    start_time = time.time()
    results['simulated_annealing'] = classical_opt.simulated_annealing(
        start_position, initial_temp=50.0, cooling_rate=0.98
    )
    results['simulated_annealing']['time'] = time.time() - start_time
    print(f"   ✓ Completed in {results['simulated_annealing']['time']:.3f}s")
    print(f"   Final energy: {results['simulated_annealing']['final_energy']:.4f}")
    print(f"   Acceptance rate: {results['simulated_annealing']['acceptance_rate']:.2%}")
    
    # Quantum-enhanced optimization
    print("\n3. Running Quantum-Enhanced Gradient Descent...")
    tunneling_sim = QuantumTunnelingSimulator(mass=1.67e-27)
    quantum_opt = QuantumEnhancedOptimizer(landscape, tunneling_sim)
    start_time = time.time()
    results['quantum_enhanced'] = quantum_opt.quantum_gradient_descent(
        start_position, learning_rate=0.15, tunnel_threshold=3.0
    )
    results['quantum_enhanced']['time'] = time.time() - start_time
    print(f"   ✓ Completed in {results['quantum_enhanced']['time']:.3f}s")
    print(f"   Final energy: {results['quantum_enhanced']['final_energy']:.4f}")
    print(f"   Tunneling events: {results['quantum_enhanced']['num_tunneling']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Final Energy':<15} {'Time (s)':<12} {'Iterations'}")
    print("-" * 70)
    for method_name, method_results in results.items():
        method_label = method_name.replace('_', ' ').title()
        print(f"{method_label:<30} {method_results['final_energy']:>10.4f}     "
              f"{method_results['time']:>7.3f}     {method_results['iterations']:>6}")
    
    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]['final_energy'])
    print("-" * 70)
    print(f"🏆 Best result: {best_method[0].replace('_', ' ').title()} "
          f"(Energy: {best_method[1]['final_energy']:.4f})")
    
    return results


def visualize_optimization_comparison(landscape, results):
    """
    Create comprehensive visualization of optimization results.
    """
    # Generate landscape
    X, Y, Z = landscape.generate_2d_landscape((0, 8), (0, 8), resolution=100)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Energy landscape with all trajectories
    ax1 = fig.add_subplot(231)
    contour = ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, Z, levels=15, colors='white', alpha=0.2, linewidths=0.5)
    
    colors = {'gradient_descent': 'red', 'simulated_annealing': 'blue', 'quantum_enhanced': 'lime'}
    labels = {'gradient_descent': 'Gradient Descent', 'simulated_annealing': 'Simulated Annealing', 
              'quantum_enhanced': 'Quantum Enhanced'}
    
    for method, result in results.items():
        traj = result['trajectory']
        ax1.plot(traj[:, 0], traj[:, 1], color=colors[method], linewidth=2, 
                label=labels[method], alpha=0.8)
        ax1.plot(traj[0, 0], traj[0, 1], 'wo', markersize=10, markeredgecolor='black', 
                markeredgewidth=2, label='Start' if method == 'gradient_descent' else '')
        ax1.plot(traj[-1, 0], traj[-1, 1], 'o', color=colors[method], markersize=10, 
                markeredgecolor='black', markeredgewidth=2)
    
    # Mark tunneling events
    if 'quantum_enhanced' in results and results['quantum_enhanced']['num_tunneling'] > 0:
        for event in results['quantum_enhanced']['tunnel_events']:
            ax1.plot([event['from'][0], event['to'][0]], 
                    [event['from'][1], event['to'][1]], 
                    'yellow', linewidth=3, linestyle='--', alpha=0.8)
            ax1.plot(event['to'][0], event['to'][1], 'y*', markersize=20, 
                    markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Y Position', fontsize=11)
    ax1.set_title('Optimization Trajectories on Energy Landscape', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    plt.colorbar(contour, ax=ax1, label='Energy')
    
    # 2. Energy vs iterations
    ax2 = fig.add_subplot(232)
    for method, result in results.items():
        ax2.plot(result['energies'], color=colors[method], linewidth=2, label=labels[method])
        
        # Mark tunneling events
        if method == 'quantum_enhanced' and result['num_tunneling'] > 0:
            for event in result['tunnel_events']:
                ax2.axvline(x=event['iteration'], color='yellow', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Energy', fontsize=11)
    ax2.set_title('Energy Convergence', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Final energy comparison (bar chart)
    ax3 = fig.add_subplot(233)
    methods = list(results.keys())
    final_energies = [results[m]['final_energy'] for m in methods]
    method_labels = [labels[m] for m in methods]
    bars = ax3.bar(range(len(methods)), final_energies, 
                   color=[colors[m] for m in methods], edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, energy) in enumerate(zip(bars, final_energies)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(method_labels, rotation=15, ha='right')
    ax3.set_ylabel('Final Energy', fontsize=11)
    ax3.set_title('Final Energy Comparison', fontsize=12, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Convergence speed comparison
    ax4 = fig.add_subplot(234)
    iterations = [results[m]['iterations'] for m in methods]
    times = [results[m]['time'] * 1000 for m in methods]  # Convert to ms
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, iterations, width, label='Iterations', 
                    color='skyblue', edgecolor='black')
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, times, width, label='Time (ms)', 
                         color='salmon', edgecolor='black')
    
    ax4.set_xlabel('Method', fontsize=11)
    ax4.set_ylabel('Iterations', fontsize=11, color='skyblue')
    ax4_twin.set_ylabel('Time (ms)', fontsize=11, color='salmon')
    ax4.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(method_labels, rotation=15, ha='right')
    ax4.tick_params(axis='y', labelcolor='skyblue')
    ax4_twin.tick_params(axis='y', labelcolor='salmon')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. 3D trajectory visualization
    ax5 = fig.add_subplot(235, projection='3d')
    for method, result in results.items():
        traj = result['trajectory']
        energies_traj = result['energies']
        ax5.plot(traj[:, 0], traj[:, 1], energies_traj, 
                color=colors[method], linewidth=2, label=labels[method])
    
    ax5.set_xlabel('X', fontsize=10)
    ax5.set_ylabel('Y', fontsize=10)
    ax5.set_zlabel('Energy', fontsize=10)
    ax5.set_title('3D Optimization Paths', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    
    # 6. Distance from global minimum over time
    ax6 = fig.add_subplot(236)
    global_minimum = [6, 5]  # Known global minimum position
    
    for method, result in results.items():
        traj = result['trajectory']
        distances = [np.linalg.norm(pos - global_minimum) for pos in traj]
        ax6.plot(distances, color=colors[method], linewidth=2, label=labels[method])
    
    ax6.set_xlabel('Iteration', fontsize=11)
    ax6.set_ylabel('Distance from Global Minimum', fontsize=11)
    ax6.set_title('Convergence to Global Minimum', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'optimization_comparison.png'")
    plt.show()


if __name__ == "__main__":
    from quantum_tunneling_core import MolecularEnergyLandscape
    
    print("\n🚀 Starting Optimization Algorithm Demonstrations...\n")
    
    # Create test landscape
    landscape = MolecularEnergyLandscape(num_dimensions=2)
    landscape.add_binding_site(position=[2, 2], depth=-5.0, width=0.8)
    landscape.add_binding_site(position=[6, 5], depth=-8.0, width=1.0)  # Global minimum
    landscape.add_binding_site(position=[4, 7], depth=-4.0, width=0.6)
    landscape.add_barrier(position=[4, 3.5], height=6.0, width=0.8)
    landscape.add_barrier(position=[5, 6], height=3.0, width=0.5)
    
    # Starting position (near shallow local minimum)
    start_pos = [1.5, 2.5]
    
    # Run comparison
    results = compare_optimizers(landscape, start_pos)
    
    # Visualize
    visualize_optimization_comparison(landscape, results)
    
    print("\n✓ All optimizations completed successfully!")