def env_info(env):
    """Prints the information about the environment

    """
    print('-----------')
    print("Env Info")
    print('-----------')
    if env.spec is not None:
        print(env.spec.id)
    print('Action Space:', env.action_space)
    # TODO: print env_config if it exists
    # print('Observation Mode:', env.obs_mode)
    # print('Gym Observation Space:', self.genv.observation_space)
    # print('Gym Observation Space Shape:', self.genv.observation_space.shape)
    print('Observation Space:', env.observation_space)
    #if hasattr(env.observation_space, 'spaces'):
    #    print('Observation Space Spaces:',
    #          [obs for obs in env.observation_space.spaces])
    #    print('Observation Space Types:', [type(obs) for obs in env.observation_space.spaces])
    print('Reward Range:', env.reward_range)
    print('Metadata:', env.metadata)
    print('--------------------------------------')


def register_with_gym(env_id: str, entry_point: str):
    """Registers the environment with gym registry

    """
    from gym.envs.registration import register, registry

    env_dict = registry.env_specs.copy()
    for env in env_dict:
        if env_id in env:
            print(f"Removing {env} from Gym registry")
            del registry.env_specs[env]

    print(f"navsim_envs: Adding {env_id} to Gym registry")
    return register(id=env_id, entry_point=entry_point)

def env_state_shapes(env):
    return [obs.shape for obs in env.observation_space.spaces] \
        if hasattr(env,'spaces') else [env.observation_space.shape]
