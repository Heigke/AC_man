from gym.envs.registration import register

register(
    id='AC_man-v0',
    entry_point='AC_man.AC_man:Automatic_Control_Environment',
)