from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces


class MultiTask(MiniGridEnv):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(
        self,
        size=16,
        task: int | None = None,  # None for all tasks
        subtasks: list[str] = [],  # string of pattern
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        random_goal: bool = False,
        noise_prob: float = 0.0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.random_goal = random_goal
        self.noise_prob = noise_prob
        self.task = task
        self.subtasks = subtasks

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 2 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        self.action_space = spaces.Discrete(3)
        self.action_history = ""
        self.reward_dimension = len(self.subtasks) + 1
        self.cd_flag = [0] * (self.reward_dimension - 1)

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        if self.random_goal:
            self.goal_pos = (
                self._rand_int(1, width - 2),
                self._rand_int(1, height - 2),
            )
        else:
            self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self, **kwargs):
        self.action_history = ""
        if self.random_goal:
            self.goal_pos = (
                self._rand_int(1, self.width - 2),
                self._rand_int(1, self.height - 2),
            )
        obs = super().reset(**kwargs)
        return obs

    def step(self, action):
        if self._rand_float(0, 1) < self.noise_prob:
            action_candidate = [i for i in range(3) if i != action]
            new_action = self._rand_subset(action_candidate, 1)[0]
            # print(f"{action} -> {new_action}")
            action = new_action
        self.action_history += str(action)

        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        # done = self.step_count >= 50

        rewards = [0] * self.reward_dimension
        rewards[0] = reward
        for i in range(1, self.reward_dimension):
            if (
                self.subtasks[i - 1]
                == self.action_history[-len(self.subtasks[i - 1]) :]
            ):
                rewards[i] = 1
            # if self.cd_flag[i - 1] != 0:
            #     self.cd_flag[i - 1] -= 1
            #     continue
            # elif (
            #     self.subtasks[i - 1]
            #     == self.action_history[-len(self.subtasks[i - 1]) :]
            # ):
            #     rewards[i] = 1
            #     self.cd_flag[i - 1] = len(self.subtasks[i - 1]) - 1

        if self.task is not None:
            rewards = rewards[self.task]

        return obs, rewards, terminated, truncated, info
