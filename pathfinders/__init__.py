from gymnasium.envs.registration import register


register(
    id='castle-v0',
    entry_point="pathfinders.envs.castle:CastleEnv"
)

register(
    id='shortcutmaze-v0',
    entry_point="pathfinders.envs.shortcutmaze:ShortcutMazeEnv"
)

register(
    id='shortcutshortmaze-v0',
    entry_point="pathfinders.envs.shortcutmaze_short:ShortcutShortMazeEnv"
)

register(
    id='city-v0',
    entry_point="pathfinders.envs.city:CityEnv"
)

