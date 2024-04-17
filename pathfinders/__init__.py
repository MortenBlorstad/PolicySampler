from gymnasium.envs.registration import register


register(
    id='castle-v0',
    entry_point="pathfinders.envs.castle:CastleEnv"
)
