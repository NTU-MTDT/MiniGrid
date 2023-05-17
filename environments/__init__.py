from .multitask import MultiTask

from gymnasium.envs.registration import register

register(
    id="SingleTaskGoal-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 16,
        "task": 0,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": None,
    },
)

register(
    id="SingleTaskRight-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 16,
        "task": 1,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": None,
    },
)

register(
    id="SingleTaskLeft-v0",
    entry_point=MultiTask,
    kwargs={
        "size": 16,
        "task": 2,
        "subtasks": ["1111", "0000"],
        "agent_start_pos": None,
    },
)

register(
    id="SingleTask-v0",
    entry_point=MultiTask,
    kwargs={"size": 16, "agent_start_pos": None},
)

register(
    id="ThreeTask-v0",
    entry_point=MultiTask,
    kwargs={"size": 16, "subtasks": ["1111", "0000"], "agent_start_pos": None},
)
