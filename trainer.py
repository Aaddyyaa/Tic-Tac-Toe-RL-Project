# trainer.py
# =====================================================================
# Training system with history tracking and visualization
# =====================================================================

import matplotlib.pyplot as plt
from game_manager import GameManager
from agents import QLearningAgent, SARSAAgent, RandomAgent, ScriptedAgent


class Trainer:
    """Trains a single agent pair and tracks performance history."""
    
    def __init__(self, agent_x, agent_o, episodes=10000, checkpoint_interval=500):
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.episodes = episodes
        self.checkpoint_interval = checkpoint_interval
        
        # Track results
        self.results = {"X_wins": 0, "O_wins": 0, "Draws": 0}
        
        # Track history for plotting
        self.history = {
            "episodes": [],
            "x_win_rate": [],
            "o_win_rate": [],
            "draw_rate": []
        }

    def train(self):
        """Trains agents over multiple episodes."""
        gm = GameManager(self.agent_x, self.agent_o, verbose=False)
        
        temp_wins_x = 0
        temp_wins_o = 0
        temp_draws = 0

        for episode in range(1, self.episodes + 1):
            result = gm.play_game(train=True)

            if result == 1:
                self.results["X_wins"] += 1
                temp_wins_x += 1
            elif result == -1:
                self.results["O_wins"] += 1
                temp_wins_o += 1
            else:
                self.results["Draws"] += 1
                temp_draws += 1

            # Record checkpoint
            if episode % self.checkpoint_interval == 0:
                total = temp_wins_x + temp_wins_o + temp_draws
                self.history["episodes"].append(episode)
                self.history["x_win_rate"].append(100 * temp_wins_x / total)
                self.history["o_win_rate"].append(100 * temp_wins_o / total)
                self.history["draw_rate"].append(100 * temp_draws / total)
                
                # Reset temp counters
                temp_wins_x = temp_wins_o = temp_draws = 0
                
                print(f"Episode {episode}/{self.episodes} completed...")

        return self.results

    def get_statistics(self):
        """Returns performance statistics."""
        total = sum(self.results.values())
        return {
            "total_games": total,
            "x_wins": self.results["X_wins"],
            "o_wins": self.results["O_wins"],
            "draws": self.results["Draws"],
            "x_win_rate": 100 * self.results["X_wins"] / total if total > 0 else 0,
            "o_win_rate": 100 * self.results["O_wins"] / total if total > 0 else 0,
            "draw_rate": 100 * self.results["Draws"] / total if total > 0 else 0
        }


class TrainingOrchestrator:
    """Orchestrates training of multiple agents and comparison."""
    
    def __init__(self, episodes_per_training=10000):
        self.episodes_per_training = episodes_per_training
        self.training_results = {}
        self.all_histories = {}
        self.trained_agents = {}
        
    def train_all_agents(self):
        """Train Q-Learning and SARSA agents against Random and Scripted opponents."""
        
        print("=" * 70)
        print("TRAINING ALL AGENTS")
        print("=" * 70)
        
        # Create learning agents
        q_vs_random = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2, name="Q-Learning-vs-Random")
        q_vs_scripted = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2, name="Q-Learning-vs-Scripted")
        sarsa_vs_random = SARSAAgent(alpha=0.1, gamma=0.9, epsilon=0.2, name="SARSA-vs-Random")
        sarsa_vs_scripted = SARSAAgent(alpha=0.1, gamma=0.9, epsilon=0.2, name="SARSA-vs-Scripted")
        
        # Store trained agents
        self.trained_agents = {
            "Q-Learning (trained vs Random)": q_vs_random,
            "Q-Learning (trained vs Scripted)": q_vs_scripted,
            "SARSA (trained vs Random)": sarsa_vs_random,
            "SARSA (trained vs Scripted)": sarsa_vs_scripted
        }
        
        # Training configurations
        trainings = [
            (q_vs_random, RandomAgent("Random"), "Q-Learning vs Random"),
            (q_vs_scripted, ScriptedAgent("Scripted"), "Q-Learning vs Scripted"),
            (sarsa_vs_random, RandomAgent("Random"), "SARSA vs Random"),
            (sarsa_vs_scripted, ScriptedAgent("Scripted"), "SARSA vs Scripted")
        ]
        
        # Train each configuration
        for learning_agent, opponent, name in trainings:
            print(f"\n{'='*70}")
            print(f"Training: {name}")
            print(f"{'='*70}")
            
            trainer = Trainer(learning_agent, opponent, 
                            episodes=self.episodes_per_training, 
                            checkpoint_interval=1000)
            trainer.train()
            
            stats = trainer.get_statistics()
            self.training_results[name] = stats
            self.all_histories[name] = trainer.history
            
            print(f"\nResults for {name}:")
            print(f"  Wins: {stats['x_wins']} ({stats['x_win_rate']:.2f}%)")
            print(f"  Losses: {stats['o_wins']} ({stats['o_win_rate']:.2f}%)")
            print(f"  Draws: {stats['draws']} ({stats['draw_rate']:.2f}%)")
        
        return self.trained_agents
    
    def plot_training_progress(self):
        """Plot win rates over time for all training sessions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress: Win Rates Over Time', fontsize=16, fontweight='bold')
        
        training_names = list(self.all_histories.keys())
        
        for idx, (name, history) in enumerate(self.all_histories.items()):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            ax.plot(history['episodes'], history['x_win_rate'], 
                   label='Learning Agent Win Rate', linewidth=2, marker='o')
            ax.plot(history['episodes'], history['o_win_rate'], 
                   label='Opponent Win Rate', linewidth=2, marker='s')
            ax.plot(history['episodes'], history['draw_rate'], 
                   label='Draw Rate', linewidth=2, marker='^')
            
            ax.set_xlabel('Episodes', fontsize=11)
            ax.set_ylabel('Rate (%)', fontsize=11)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        print("\n✓ Training progress plot saved as 'training_progress.png'")
        plt.show()
    
    def display_final_comparison(self):
        """Display a comparison of all trained agents."""
        print("\n" + "=" * 70)
        print("FINAL PERFORMANCE COMPARISON")
        print("=" * 70)
        
        for name, stats in self.training_results.items():
            print(f"\n{name}:")
            print(f"  Total Games: {stats['total_games']}")
            print(f"  Win Rate:    {stats['x_win_rate']:.2f}%")
            print(f"  Loss Rate:   {stats['o_win_rate']:.2f}%")
            print(f"  Draw Rate:   {stats['draw_rate']:.2f}%")
        
        # Determine best performer
        print("\n" + "=" * 70)
        best_agent = max(self.training_results.items(), 
                        key=lambda x: x[1]['x_win_rate'])
        print(f"🏆 BEST PERFORMER: {best_agent[0]}")
        print(f"   Win Rate: {best_agent[1]['x_win_rate']:.2f}%")
        print("=" * 70)