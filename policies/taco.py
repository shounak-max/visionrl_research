"""
TACO (Task-Aligned Contrastive RL) Policy.

Extends DrQ-v2 with a temporal contrastive auxiliary loss:
- Anchor embedding z_t = encoder(obs_t)
- Positive embedding z_{t+k} predicted via learned transition model h(z_t, a_t)
- InfoNCE contrastive loss on (z_t, predicted_z_{t+1}, negatives_from_batch)
- Total loss = actor-critic loss + beta * contrastive_loss

Reference: Zheng et al., "TACO: Temporal Latent Action-Driven Contrastive Loss
for Visual Reinforcement Learning", NeurIPS 2023.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.drq_v2 import DrQv2Policy


class TACOPolicy(DrQv2Policy):
    """
    TACO policy: DrQ-v2 + temporal contrastive auxiliary loss.

    Additional components:
      - projection_head: maps embedding_dim -> 128 (for contrastive space)
      - transition_model: predicts z_{t+1} from (z_t, a_t)
    """

    def __init__(
        self,
        action_dim: int = 2,
        embedding_dim: int = 512,
        is_continuous: bool = True,
        lr: float = 1e-4,
        gamma: float = 0.99,
        beta: float = 0.1,
        proj_dim: int = 128,
    ):
        super().__init__(
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            is_continuous=is_continuous,
            lr=lr,
            gamma=gamma,
        )
        self.beta = beta
        self.proj_dim = proj_dim

        # Projection head: 2-layer MLP, embedding_dim -> 256 -> proj_dim
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

        # Transition model: (z_t, a_t) -> z_{t+1} in embedding space
        self.transition_model = nn.Linear(embedding_dim + action_dim, embedding_dim)

        # Re-create optimizer to include the new parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def temporal_contrastive_loss(
        self,
        z_anchor: torch.Tensor,
        z_positive_predicted: torch.Tensor,
        all_embeddings: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Args:
            z_anchor: [B, proj_dim]  — projected anchor embeddings z_t
            z_positive_predicted: [B, proj_dim] — projected predicted z_{t+1}
            all_embeddings: [B, proj_dim] — all projected embeddings (negatives are
                                             all other items in the batch)
            temperature: softmax temperature

        Returns:
            Scalar InfoNCE loss.
        """
        # Normalize all to unit sphere
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_pos = F.normalize(z_positive_predicted, dim=-1)
        all_emb = F.normalize(all_embeddings, dim=-1)  # [B, proj_dim]

        # Similarity of anchor to all embeddings: [B, B]
        sim_matrix = torch.matmul(z_anchor, all_emb.T) / temperature  # [B, B]

        # The positive for each anchor i is z_pos[i]; compute anchor-positive sim
        # We use a separate positive that is NOT in all_emb (the predicted future)
        pos_sim = (z_anchor * z_pos).sum(dim=-1, keepdim=True) / temperature  # [B, 1]

        # Concatenate positive similarity with all pairwise similarities to get logits
        # Logits: [B, B+1], where column 0 is the positive
        logits = torch.cat([pos_sim, sim_matrix], dim=1)  # [B, B+1]

        # Labels: index 0 is the positive for every row
        labels = torch.zeros(z_anchor.shape[0], dtype=torch.long, device=z_anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        TACO update: DrQ-v2 actor-critic loss + beta * temporal contrastive loss.
        """
        self.train()
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        # 1. Augment observations
        obs_p = self.encoder.preprocess_obs(obs)
        next_obs_p = self.encoder.preprocess_obs(next_obs)
        obs_aug = self.aug(obs_p)
        next_obs_aug = self.aug(next_obs_p)

        # 2. Compute anchor and actual next embeddings
        emb = self.encoder.embed(obs_aug)           # z_t: [B, embedding_dim]
        with torch.no_grad():
            next_emb = self.encoder.embed(next_obs_aug)  # z_{t+1}: [B, embedding_dim]
            if self.is_continuous:
                next_action = self.actor(next_emb)
                tq1 = self.target_q1(torch.cat([next_emb, next_action], dim=-1))
                tq2 = self.target_q2(torch.cat([next_emb, next_action], dim=-1))
                target_q = torch.min(tq1, tq2)
                target_value = reward + (1.0 - done) * self.gamma * target_q
            else:
                nq1 = self.target_q1(next_emb)
                nq2 = self.target_q2(next_emb)
                target_q = torch.min(nq1, nq2).max(dim=-1, keepdim=True)[0]
                target_value = reward + (1.0 - done) * self.gamma * target_q

        # 3. Critic loss
        if self.is_continuous:
            current_q1 = self.q1(torch.cat([emb, action], dim=-1))
            current_q2 = self.q2(torch.cat([emb, action], dim=-1))
            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
        else:
            act_idx = action.view(-1, 1).long()
            current_q1 = self.q1(emb).gather(1, act_idx)
            current_q2 = self.q2(emb).gather(1, act_idx)
            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        # 4. Actor loss
        if self.is_continuous:
            pred_action = self.actor(emb.detach())
            actor_loss = -self.q1(torch.cat([emb.detach(), pred_action], dim=-1)).mean()
        else:
            logits = self.actor(emb.detach())
            log_probs = F.log_softmax(logits, dim=-1)
            q_vals = self.q1(emb.detach())
            actor_loss = -(log_probs * F.softmax(q_vals, dim=-1)).sum(dim=-1).mean()

        # 5. TACO contrastive loss
        # Predict z_{t+1} from (z_t, a_t) using the transition model
        z_predicted = self.transition_model(
            torch.cat([emb.detach(), action], dim=-1)
        )  # [B, embedding_dim]

        # Project to contrastive space
        z_anchor_proj = self.projection_head(emb.detach())          # [B, proj_dim]
        z_pos_proj = self.projection_head(z_predicted)               # [B, proj_dim]
        # All next embeddings (from the batch) serve as the full contrast set
        z_all_proj = self.projection_head(next_emb.detach())        # [B, proj_dim]

        contrastive_loss = self.temporal_contrastive_loss(
            z_anchor_proj, z_pos_proj, z_all_proj
        )

        # 6. Combined loss
        total_loss = critic_loss + actor_loss + self.beta * contrastive_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Polyak update target networks
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.target_q1.parameters()):
                pt.data.mul_(1.0 - self.tau)
                pt.data.add_(self.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.target_q2.parameters()):
                pt.data.mul_(1.0 - self.tau)
                pt.data.add_(self.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "contrastive_loss": float(contrastive_loss.item()),
            "total_loss": float(total_loss.item()),
        }
