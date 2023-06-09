import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):
    def train_step(self):
        # s, a, r, d, rtg, timesteps, task_maks, attention_mask
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            task_masks,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            task_masks,
            attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        return_dim = reward_preds.shape[2]
        reward_preds = reward_preds.reshape(-1, return_dim)[
            attention_mask.reshape(-1) > 0
        ]
        reward_target = rewards.reshape(-1, return_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None,
            action_preds,
            reward_preds,
            None,
            action_target,
            reward_target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        # log the output accuracy
        # using argmax to get the index of the max log-probability
        with torch.no_grad():
            self.diagnostics["training/action_error"] = np.mean(
                np.argmax(action_preds.detach().cpu().numpy(), axis=1)
                != np.argmax(action_target.detach().cpu().numpy(), axis=1)
            )
            # self.diagnostics['training/action_error'] = torch.mean().detach().cpu().item()

        return loss.detach().cpu().item()
