from .multitask import MultiTask
from minigrid.core.world_object import Floor, Lava

from gymnasium.envs.registration import register

register(
    id="SingleTaskGoal-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 9,
        "task": 0,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": None,
    },
)

register(
    id="SingleTaskRight-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 9,
        "task": 1,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": None,
    },
)

register(
    id="SingleTaskLeft-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 9,
        "task": 2,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": None,
    },
)

register(
    id="SingleTaskLava-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 9,
        "task": 1,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": None,
        "obstacle_type": Lava,
    },
)

register(
    id="SingleTask-v0",
    entry_point=MultiTask,
    kwargs={"size": 9, "agent_start_pos": (1, 1)},
)

register(
    id="LavaGoal-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 9,
        "agent_start_pos": (1, 1),
        "obstacle_type": Lava,
    },
)

register(
    id="ThreeTask-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 9,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": (1, 1),
        "max_steps": 100,
    },
)
