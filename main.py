# main.py
# =====================================================================
# Main execution file for Tic-Tac-Toe RL Training System
# =====================================================================

from trainer import TrainingOrchestrator
from game_manager import GameManager
from agents import HumanAgent


def play_against_agent(agent):
    """Allow human to play against a trained agent."""
    print("\n" + "=" * 70)
    print(f"PLAYING AGAINST: {agent.name}")
    print("=" * 70)
    print("\nYou are O, the agent is X. Board positions:")
    print("0 | 1 | 2")
    print("--+---+--")
    print("3 | 4 | 5")
    print("--+---+--")
    print("6 | 7 | 8")
    
    human = HumanAgent("You")
    gm = GameManager(agent, human, verbose=True)
    result = gm.play_game(train=False)
    
    if result == 1:
        print(f"\n{agent.name} wins!")
    elif result == -1:
        print("\nYou win! Congratulations!")
    else:
        print("\nIt's a draw!")


def evaluate_agents(trained_agents, episodes=1000):
    """Allow evaluator to pit any two trained agents against each other."""
    print("\n" + "=" * 70)
    print("AGENT VS AGENT EVALUATION")
    print("=" * 70)
    
    agent_list = list(trained_agents.items())
    
    print("\nAvailable trained agents:")
    for idx, (name, _) in enumerate(agent_list, 1):
        print(f"{idx}. {name}")
    
    try:
        print("\n--- Select Agent X (plays first) ---")
        choice_x = int(input("Enter choice: "))
        if not (1 <= choice_x <= len(agent_list)):
            print("Invalid choice!")
            return
        
        print("\n--- Select Agent O (plays second) ---")
        choice_o = int(input("Enter choice: "))
        if not (1 <= choice_o <= len(agent_list)):
            print("Invalid choice!")
            return
        
        agent_x_name, agent_x = agent_list[choice_x - 1]
        agent_o_name, agent_o = agent_list[choice_o - 1]
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {agent_x_name} (X) vs {agent_o_name} (O)")
        print(f"Playing {episodes} games...")
        print(f"{'='*70}\n")
        
        # Run evaluation games
        gm = GameManager(agent_x, agent_o, verbose=False)
        results = {"X_wins": 0, "O_wins": 0, "Draws": 0}
        
        for i in range(episodes):
            result = gm.play_game(train=False)  # No training during evaluation
            if result == 1:
                results["X_wins"] += 1
            elif result == -1:
                results["O_wins"] += 1
            else:
                results["Draws"] += 1
            
            if (i + 1) % (episodes // 10) == 0:
                print(f"Progress: {i + 1}/{episodes} games completed...")
        
        # Display results
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"\n{agent_x_name} (X):")
        print(f"  Wins: {results['X_wins']} ({100 * results['X_wins'] / episodes:.2f}%)")
        print(f"\n{agent_o_name} (O):")
        print(f"  Wins: {results['O_wins']} ({100 * results['O_wins'] / episodes:.2f}%)")
        print(f"\nDraws: {results['Draws']} ({100 * results['Draws'] / episodes:.2f}%)")
        
        # Determine winner
        if results['X_wins'] > results['O_wins']:
            print(f"\n🏆 Winner: {agent_x_name}")
        elif results['O_wins'] > results['X_wins']:
            print(f"\n🏆 Winner: {agent_o_name}")
        else:
            print(f"\n🤝 It's a tie!")
        print(f"{'='*70}\n")
        
    except ValueError:
        print("Please enter valid numbers!")
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled.")


def main():
    """Main execution function."""
    
    # Train all agents
    print("Starting training process...")
    print("This will train 4 agents (40,000 total games)")
    print()
    
    orchestrator = TrainingOrchestrator(episodes_per_training=10000)
    trained_agents = orchestrator.train_all_agents()
    
    # Show comparison
    orchestrator.display_final_comparison()
    
    # Plot training progress
    orchestrator.plot_training_progress()
    
    # Main menu
    print("\n" + "=" * 70)
    print("MAIN MENU")
    print("=" * 70)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Play against a trained agent")
        print("2. Evaluate: Agent vs Agent")
        print("3. Exit")
        
        try:
            main_choice = int(input("\nEnter your choice: "))
            
            if main_choice == 1:
                # Play against agent
                print("\n" + "=" * 70)
                print("PLAY AGAINST TRAINED AGENTS")
                print("=" * 70)
                print("\nChoose an agent to play against:")
                agent_list = list(trained_agents.items())
                for idx, (name, _) in enumerate(agent_list, 1):
                    print(f"{idx}. {name}")
                
                choice = int(input("\nEnter your choice: "))
                if 1 <= choice <= len(agent_list):
                    agent_name, agent = agent_list[choice - 1]
                    play_against_agent(agent)
                else:
                    print("Invalid choice!")
            
            elif main_choice == 2:
                # Agent vs Agent evaluation
                evaluate_agents(trained_agents, episodes=1000)
            
            elif main_choice == 3:
                print("\nThanks for using the Tic-Tac-Toe RL Training System!")
                break
            
            else:
                print("Invalid choice! Please enter 1, 2, or 3.")
                
        except ValueError:
            print("Please enter a valid number!")
        except KeyboardInterrupt:
            print("\n\nThanks for using the system!")
            break


if __name__ == "__main__":
    main()