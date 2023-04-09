from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace


class EmptyEnv(MiniGridEnv):

    """
    ### Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ### Mission Space

    "get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    """

    def __init__(self, size=10, agent_start_pos=None, agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

        self.action_history = ""
        self.length = 0
        self.reward_dimension = 2
        self.subtasks = ["0"]
        self.cd_flag = [0] * (self.reward_dimension - 1)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def step(self, action):
        obs, reward, done, info = super().step(action)
        done = self.step_count >= 50

        self.action_history += str(action)

        rewards = [0] * self.reward_dimension
        rewards[0] = reward
        for i in range(1, self.reward_dimension):
            if self.cd_flag[i - 1] != 0:
                self.cd_flag[i - 1] -= 1
                continue
            elif (
                self.subtasks[i - 1]
                == self.action_history[-len(self.subtasks[i - 1]) :]
            ):
                rewards[i] = 1
                self.cd_flag[i - 1] = len(self.subtasks[i - 1]) - 1

        return obs, rewards, done, info
