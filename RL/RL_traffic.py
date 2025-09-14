
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
import random
from collections import deque
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class Config:
    """Configuration class for easy parameter management"""
    num_junctions: int = 50
    num_parameters: int = 5  # a, b, c, d, e
    state_dim: int = 5
    action_dim: int = 1  # Binary action (0, 1)
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    eps_clip: float = 0.2
    k_coeff: float = 100.0  # K constant in reward function
    batch_size: int = 64
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    update_interval: int = 10

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer to handle junction positions"""
    
    def _init_(self, d_model: int, max_len: int = 5000):
        super()._init_()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerAgent(nn.Module):
    """Transformer-based agent for traffic management with parameter sharing"""
    
    def _init_(self, config: Config):
        super()._init_()
        self.config = config
        
        # Input projection layer
        self.input_projection = nn.Linear(config.state_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.num_junctions)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Output heads for policy and value
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.action_dim),
            nn.Sigmoid()  # For binary actions (0, 1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, states):
        """
        Forward pass through transformer
        Args:
            states: [batch_size, num_junctions, state_dim]
        Returns:
            action_probs: [batch_size, num_junctions, action_dim]
            values: [batch_size, num_junctions, 1]
        """
        batch_size, num_junctions, state_dim = states.shape
        
        # Project input to hidden dimension
        x = self.input_projection(states)  # [batch_size, num_junctions, hidden_dim]
        
        # Add positional encoding
        x = x.permute(1, 0, 2)  # [num_junctions, batch_size, hidden_dim]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # [batch_size, num_junctions, hidden_dim]
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(x)  # [batch_size, num_junctions, hidden_dim]
        
        # Generate policy and value outputs
        action_probs = self.policy_head(transformer_output)  # [batch_size, num_junctions, action_dim]
        values = self.value_head(transformer_output)  # [batch_size, num_junctions, 1]
        
        return action_probs, values

class TrafficEnvironment:
    """Simulated traffic environment with configurable junctions and parameters"""
    
    def _init_(self, config: Config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize random state for each junction
        # State format: [a, b, c, d, e] for each junction
        self.states = np.random.uniform(0, 100, (self.config.num_junctions, self.config.num_parameters))
        self.step_count = 0
        return self.get_states()
    
    def get_states(self):
        """Get current states maintaining junction individuality"""
        return torch.FloatTensor(self.states).unsqueeze(0)  # [1, num_junctions, state_dim]
    
    def step(self, actions):
        """
        Execute actions and return new state, reward, done
        Args:
            actions: [num_junctions] binary actions (0 or 1)
        """
        actions = actions.cpu().numpy().flatten()
        
        # Simulate state transitions based on actions
        # Simple dynamics: actions affect the state parameters
        for i, action in enumerate(actions):
            if action == 1:
                # Action 1: Reduce congestion (parameter 'a')
                self.states[i, 0] = max(0, self.states[i, 0] - np.random.uniform(1, 5))
                # Side effects on other parameters
                self.states[i, 1:] += np.random.uniform(-2, 2, 4)
            else:
                # Action 0: Natural evolution
                self.states[i, 0] += np.random.uniform(0, 3)
                self.states[i, 1:] += np.random.uniform(-1, 1, 4)
        
        # Add noise and ensure bounds
        self.states = np.clip(self.states + np.random.normal(0, 0.5, self.states.shape), 0, 200)
        
        # Calculate reward: k - sigma(a)/50
        reward = self.config.k_coeff - np.sum(self.states[:, 0]) / self.config.num_junctions
        
        self.step_count += 1
        done = self.step_count >= self.config.max_steps_per_episode
        
        return self.get_states(), reward, done

class PPOTrainer:
    """PPO trainer for multi-agent traffic management"""
    
    def _init_(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize agent and environment
        self.agent = TransformerAgent(config).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate)
        self.env = TrafficEnvironment(config)
        
        # Storage for experience
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def collect_experience(self):
        """Collect experience for one episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.max_steps_per_episode):
            state = state.to(self.device)
            
            with torch.no_grad():
                action_probs, values = self.agent(state)
            
            # Sample actions using Bernoulli distribution
            dist = Bernoulli(action_probs.squeeze(0))  # Remove batch dimension
            actions = dist.sample()  # [num_junctions, action_dim]
            log_probs = dist.log_prob(actions)
            
            # Execute actions in environment
            next_state, reward, done = self.env.step(actions)
            
            # Store experience
            self.memory['states'].append(state.cpu())
            self.memory['actions'].append(actions.cpu())
            self.memory['rewards'].append(reward)
            self.memory['log_probs'].append(log_probs.sum().cpu())  # Sum across junctions
            self.memory['values'].append(values.mean().cpu())  # Average across junctions
            self.memory['dones'].append(done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return episode_reward, episode_length
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update_agent(self):
        """Update agent using PPO algorithm"""
        if len(self.memory['states']) == 0:
            return
        
        # Convert memory to tensors
        states = torch.cat(self.memory['states']).to(self.device)
        actions = torch.cat(self.memory['actions']).to(self.device)
        old_log_probs = torch.stack(self.memory['log_probs']).to(self.device)
        values = torch.stack(self.memory['values']).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        dones = torch.FloatTensor(self.memory['dones']).to(self.device)
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards.cpu().tolist(), values.cpu().tolist(), dones.cpu().tolist())
        advantages = advantages.to(self.device)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Get current policy
            action_probs, new_values = self.agent(states)
            
            # Reshape for loss computation
            batch_size = states.shape[0]
            action_probs = action_probs.view(batch_size, -1)  # [batch_size, num_junctions * action_dim]
            actions_flat = actions.view(batch_size, -1)
            
            # Compute new log probabilities
            dist = Bernoulli(action_probs)
            new_log_probs = dist.log_prob(actions_flat).sum(dim=1)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            new_values = new_values.mean(dim=1).squeeze()  # Average across junctions
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        for key in self.memory:
            self.memory[key] = []
    
    def train(self):
        """Main training loop"""
        print(f"Starting training with {self.config.num_junctions} junctions")
        print(f"State parameters: {self.config.num_parameters} (a, b, c, d, e)")
        print(f"Using device: {self.device}")
        print("-" * 50)
        
        for episode in range(self.config.max_episodes):
            # Collect experience
            episode_reward, episode_length = self.collect_experience()
            
            # Update agent
            if episode % self.config.update_interval == 0:
                self.update_agent()
            
            # Print progress
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                print(f"Episode {episode:4d} | Avg Reward: {avg_reward:8.2f} | Episode Length: {episode_length:3d}")
        
        print("\nTraining completed!")
        return self.episode_rewards
    
    def plot_training_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Moving average
        if len(self.episode_rewards) > 10:
            window_size = min(50, len(self.episode_rewards) // 10)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.episode_rewards)), moving_avg, 'r-', alpha=0.7, label=f'Moving Avg ({window_size})')
            plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def test_agent(self, num_episodes=5):
        """Test the trained agent"""
        print(f"\nTesting agent for {num_episodes} episodes...")
        test_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config.max_steps_per_episode):
                state = state.to(self.device)
                
                with torch.no_grad():
                    action_probs, _ = self.agent(state)
                
                # Use deterministic actions for testing
                actions = (action_probs.squeeze(0) > 0.5).float()
                
                next_state, reward, done = self.env.step(actions)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        avg_test_reward = np.mean(test_rewards)
        print(f"Average Test Reward: {avg_test_reward:.2f}")
        return test_rewards

def main():
    """Main function to run the traffic management RL system"""
    # Create configuration
    config = Config(
        num_junctions=50,
        num_parameters=5,
        max_episodes=500,
        learning_rate=3e-4,
        update_interval=5
    )
    
    print("Multi-Agent Traffic Management with Transformer Networks")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Junctions: {config.num_junctions}")
    print(f"  State parameters: {config.num_parameters} (a, b, c, d, e)")
    print(f"  Action space: Binary (0, 1)")
    print(f"  Reward function: k - σ(a)/N where k={config.k_coeff}")
    print(f"  Algorithm: PPO with Transformer networks")
    print(f"  Parameter sharing: Enabled")
    print("=" * 60)
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Train the agent
    episode_rewards = trainer.train()
    
    # Plot results
    trainer.plot_training_progress()
    
    # Test the trained agent
    test_rewards = trainer.test_agent()
    
    return trainer, episode_rewards, test_rewards

if _name_ == "_main_":
    trainer, episode_rewards, test_rewards = main()
