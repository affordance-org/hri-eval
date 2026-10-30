"""The smallest policy: GR00T balance with a zero walking command, arms held at the start pose.
python run.py trials/stand_still --policy examples.stand_still:StandStill"""
import numpy as np

import episode as E


class StandStill:
    def __init__(self):
        self.loco = E.Gr00tLocomotion()

    def reset(self, ctx):
        self.loco.reset()
        self.arm = ctx.q29[15:].copy()

    def policy_step(self, ctx):
        q15 = self.loco.step(ctx, 0.0, 0.0, 0.0, base_height=0.74)
        q, kp, kd, g = E.compose_gr00t(q15, self.arm)
        return E.ControlOutput(q_target=q, kp=kp, kd=kd, gravcomp=g, fingers=np.zeros(4), cmd=self.loco.cmd,
                               policy_obs=self.loco.obs, policy_action=self.loco.action, phase='stand')

    def policy_info(self):
        return self.loco.info()

    def describe(self):
        return 'GR00T WBC Balance with a zero command; arms hold the start pose; grippers open.'

    def disclosures(self):
        return []
