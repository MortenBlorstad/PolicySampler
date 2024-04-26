from gymnasium.envs.registration import register


register(
    id='castle-v0',
    entry_point="pathfinders.envs.castle:CastleEnv"
)

register(
    id='shortcutmaze-v0',
    entry_point="pathfinders.envs.shortcutmaze:ShortcutMazeEnv"
)
