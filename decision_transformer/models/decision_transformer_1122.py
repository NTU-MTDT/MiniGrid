import numpy as np
import torch
import torch.nn as nn

import transformers

# from decision_transformer.models.model import TrajectoryModel


class RewardEncoder(nn.Module):
    def __init__(self, num_rws=7, embed_dim=64, mask_value=-100):
        super(RewardEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_rws = num_rws
        self.mask_value = mask_value
        self.fcs = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_rws)])
        self.activation = nn.ReLU()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, rewards):
        # if r is -100, then it is masked
        batch_size, length, _ = rewards.shape
        attn_mask = rewards == self.mask_value
        if (attn_mask + (rewards == 0)).all():
            return torch.zeros(batch_size, length, self.embed_dim).to(rewards.device)

        rewards = rewards.unsqueeze(2)
        rewards = torch.stack(
            [self.fcs[i](rewards[:, :, :, i]) for i in range(self.num_rws)], dim=-2
        )
        rewards = self.activation(rewards)

        rewards = rewards.view(-1, self.num_rws, self.embed_dim)
        attn_mask = attn_mask.view(-1, self.num_rws).unsqueeze(1)
        q = self.query.expand(rewards.shape[0], 1, self.embed_dim)

        out, att_map = self.attention(q, rewards, rewards, attn_mask=attn_mask)

        return out.view(batch_size, length, -1), att_map


class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, return_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.return_dim = return_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        return_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=False,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, return_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = transformers.GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        if kwargs.get("reward_encoder", None) == "linear":
            print("Using linear reward encoder")
            self.embed_return = nn.Linear(return_dim, hidden_size)
        else:
            print("Using MLP reward encoder")
            self.embed_return = RewardEncoder(num_rws=return_dim, embed_dim=hidden_size)
        # self.embed_return = nn.Linear(return_dim, hidden_size)
        self.embed_mask = nn.Linear(return_dim, hidden_size)

        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.act_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )
        self.predict_return = nn.Linear(hidden_size, self.return_dim)

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        task_masks,
        attention_mask=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        task_embeddings = self.embed_mask(task_masks)

        returns_embeddings = self.embed_return(returns_to_go)[0]

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        task_embeddings = task_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (
                    task_embeddings,
                    returns_embeddings,
                    state_embeddings,
                    action_embeddings,
                ),
                dim=1,
            )  # [64, 4, 20, 256]
            .permute(0, 2, 1, 3)  # [64, 20, 4, 256]
            .reshape(batch_size, 4 * seq_length, self.hidden_size)  # [64, 80, 256]
        )
        # print(stacked_inputs.shape)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack(
                (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
            )
            .permute(0, 2, 1)
            .reshape(batch_size, 4 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(
            x[:, 2]
        )  # predict next return given state and action
        state_preds = self.predict_state(
            x[:, 2]
        )  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(
        self, states, actions, rewards, returns_to_go, timesteps, task_masks, **kwargs
    ):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, self.return_dim)
        timesteps = timesteps.reshape(1, -1)
        task_masks = task_masks.reshape(1, -1, self.return_dim)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]
            task_masks = task_masks[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            self.return_dim,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
            task_masks = torch.cat(
                [
                    torch.zeros(
                        (
                            task_masks.shape[0],
                            self.max_length - task_masks.shape[1],
                            self.return_dim,
                        ),
                        device=task_masks.device,
                    ),
                    task_masks,
                ],
                dim=1,
            ).to(dtype=torch.float32)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            task_masks,
            attention_mask=attention_mask,
            **kwargs
        )

        return action_preds[0, -1]
