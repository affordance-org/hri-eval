"""Command level: walk to the table, stop in front of the backpack facing it, and stand. Arms hold the start pose,
grippers stay open. python run.py --policy examples.walk_to_table:WalkToTable"""
import math

import numpy as np

import interfaces as I


class WalkToTable(I.CommandPolicy):
    def __init__(self, standoff=0.50, v_max=0.35, wz_max=0.6, tol=0.05, yaw_tol_deg=4.0):
        self.standoff, self.v_max, self.wz_max, self.tol = standoff, v_max, wz_max, tol
        self.yaw_tol = math.radians(yaw_tol_deg)

    def reset(self, obs):
        bag = obs['objects']['backpack']['pos']
        self.goal = np.array([bag[0] - self.standoff, bag[1]])   # west of the table, facing east (+x) toward the bag
        self.goal_yaw = 0.0
        self.mode = 'turn'
        self.v = np.zeros(3)

    def act(self, obs):
        e = self.goal - obs['base_pos'][:2]
        dist = float(np.linalg.norm(e))
        yaw = obs['base_yaw']
        to_goal = I.wrap(math.atan2(e[1], e[0]) - yaw)
        to_final = I.wrap(self.goal_yaw - yaw)
        des = np.zeros(3)
        if self.mode == 'turn':                     # face the stance point
            des[2] = self.turn(to_goal)
            if abs(to_goal) < 0.1:
                self.mode = 'walk'
        if self.mode == 'walk':                     # walk to it, steering toward it; the last 15 cm without steering
            if dist > 0.15:
                des[0] = min(self.v_max, 0.8 * dist) * max(0.0, math.cos(to_goal))
                des[2] = float(np.clip(1.5 * to_goal, -self.wz_max, self.wz_max))
            else:
                c, s = math.cos(yaw), math.sin(yaw)
                des[:2] = 0.12 * np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]]) / max(dist, 1e-6)
            if dist < self.tol or obs['t'] > 15.0:
                self.mode = 'align'
        if self.mode == 'align':                    # turn in place to face the bag
            des = np.array([0.0, 0.0, self.turn(to_final)])
            if abs(to_final) < self.yaw_tol:
                self.mode = 'stand'
        if self.mode == 'stand':
            self.v[:] = 0.0
            return I.Command(phase='stand')
        lim = np.array([0.02, 0.02, 0.06])        # per 50 Hz tick: 1 m/s^2, 3 rad/s^2
        self.v += np.clip(des - self.v, -lim, lim)
        return I.Command(vx=self.v[0], vy=self.v[1], yaw_rate=self.v[2], grippers=(0.0, 0.0), phase=self.mode)

    def turn(self, err):
        return math.copysign(min(self.wz_max, max(0.15, 1.5 * abs(err))), err)

    def describe(self):
        return (f'Walks to a point {self.standoff:.2f} m west of the backpack (the TRUE bag pose at the start), turns to '
                'face it (east) and stands; arms hold the start pose, grippers open.')
