import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

MASK_NEG_LARGE = -1e9

class MaskedA2CPolicy(MultiInputActorCriticPolicy):
    """Custom A2C Policy that mutates actor logits to support Action Masking."""
    
    def _apply_mask_to_distribution(self, distribution, action_mask):
        if not (hasattr(distribution, "distribution") and hasattr(distribution.distribution, "logits")):
            return distribution

        logits = distribution.distribution.logits
        invalid = action_mask < 0.5

        # Defensive fallback: if all actions are masked as invalid, keep action 0 open
        all_invalid = invalid.all(dim=1, keepdim=True)
        if all_invalid.any():
            invalid = invalid.clone()
            invalid[all_invalid.squeeze(1), 0] = False

        # Fill invalid action logits with a highly negative value (-1e9)
        masked_logits = logits.masked_fill(invalid, MASK_NEG_LARGE)
        
        if hasattr(self.action_dist, "proba_distribution"):
            return self.action_dist.proba_distribution(action_logits=masked_logits)

        distribution.distribution.logits = masked_logits
        return distribution

    def _masked_distribution(self, obs):
        """Extracts features and applies the mask to the actor's probability distribution."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution = self._apply_mask_to_distribution(distribution, obs["action_mask"])
        return distribution, values

    def forward(self, obs, deterministic=False):
        distribution, values = self._masked_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        distribution, _ = self._masked_distribution(observation)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        distribution, values = self._masked_distribution(obs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy