from gymnasium.envs.registration import register

register(
    id="custom-roundabout-v0",
    entry_point="custom_envs.custom_roundabout:CustomRoundaboutEnv",
)
