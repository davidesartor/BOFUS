from typing import Protocol
from jaxtyping import Float
from numpy.typing import NDArray as Array
import numpy as np
from Box2D import b2Body, b2World, b2PolygonShape, b2CircleShape, b2Vec2


class TestFunction(Protocol):
    def __call__(self, x: Float[Array, "d"]) -> Float[Array, ""]: ...


class Push(TestFunction):
    def __init__(self):
        # bounds for the function
        self.lb = np.array([-5.0, -5.0, 0.0, -10.0, -10.0, -5.0, 20.0] * 2)
        self.ub = np.array([5.0, 5.0, 2.0 * np.pi, 10.0, 10.0, 5.0, 300.0] * 2)

        # starting xy locations for the two objects
        self.start1 = (0, 2)
        self.start2 = (0, -2)

        # goal xy locations for the two objects
        self.goal1 = (4, 3.5)
        self.goal2 = (-4, 3.5)

        # compute initial distance to goal for offset
        self.initial_dist = +np.linalg.norm(
            np.array(self.goal1) - np.array(self.start1)
        ) + np.linalg.norm(np.array(self.goal2) - np.array(self.start2))

        # step parameters for the physics simulation
        self.time_step = 1 / 100
        self.vel_iters = 10
        self.pos_iters = 10

        # create world and base
        self.world = b2World(gravity=(0.0, 0.0), doSleep=True)
        self.base = self.world.CreateStaticBody(
            position=(0, 0), shapes=b2PolygonShape(box=(500, 500))
        )

        # create first object to push (a box)
        self.obj1 = self.world.CreateDynamicBody(position=self.start1)
        self.obj1.CreateFixture(
            shape=b2PolygonShape(box=(0.5, 0.5)), density=0.05, friction=0.01
        )
        self.world.CreateFrictionJoint(
            bodyA=self.base, bodyB=self.obj1, maxForce=5, maxTorque=2
        )

        # create second object to push (a circle)
        self.obj2 = self.world.CreateDynamicBody(position=self.start2)
        self.obj2.CreateFixture(
            shape=b2CircleShape(radius=1), density=0.05, friction=0.01
        )
        self.world.CreateFrictionJoint(
            bodyA=self.base, bodyB=self.obj2, maxForce=5, maxTorque=2
        )

        # add first robot
        self.robot1 = self.world.CreateDynamicBody(position=(0, 0), angle=0)
        self.robot1.CreateFixture(
            shape=b2PolygonShape(box=(1.0, 0.3)), density=0.1, friction=0.1
        )
        self.world.CreateFrictionJoint(
            bodyA=self.base, bodyB=self.robot1, maxForce=2, maxTorque=2
        )

        # add second robot
        self.robot2 = self.world.CreateDynamicBody(position=(0, 0), angle=0)
        self.robot2.CreateFixture(
            shape=b2PolygonShape(box=(1.0, 0.3)), density=0.1, friction=0.1
        )
        self.world.CreateFrictionJoint(
            bodyA=self.base, bodyB=self.robot2, maxForce=2, maxTorque=2
        )

    @staticmethod
    def reset(obj: b2Body, x: float, y: float, angle: float):
        # Reset the specified object to the given state
        obj.position = (x, y)
        obj.angle = angle
        obj.linearVelocity = (0, 0)
        obj.angularVelocity = 0

    def __call__(self, x: Float[Array, "14"]) -> Float[Array, ""]:
        assert x.shape[-1] == 14, "Push function defined for 14-dimensional input"
        # denormalize input
        x = (self.ub - self.lb) * x + self.lb

        # extract parameters for the two robots
        robot1, robot2 = np.split(x, 2, axis=-1)
        x1, y1, angle1, vx1, vy1, torq1, sim_steps1 = robot1
        x2, y2, angle2, vx2, vy2, torq2, sim_steps2 = robot2

        # reset environment
        self.reset(self.obj1, *self.start1, 0.0)
        self.reset(self.obj2, *self.start2, 0.0)
        self.reset(self.robot1, float(x1), float(y1), float(angle1))
        self.reset(self.robot2, float(x2), float(y2), float(angle2))

        # simulating push with fixed direction pointing from robot location to body location
        tmax = int(max([sim_steps1, sim_steps2]))
        for t in range(tmax + 100):
            if t < sim_steps1:
                torque = self.robot1.mass * (torq1 - self.robot1.angularVelocity) * 30.0
                force = self.robot1.mass * (b2Vec2(vx1, vy1) - self.robot1.linearVelocity) * 30.0
                self.robot1.ApplyTorque(torque, wake=True)
                self.robot1.ApplyForce(force, self.robot1.position, wake=True)
            if t < sim_steps2:
                torque = self.robot2.mass * (torq2 - self.robot2.angularVelocity) * 30.0
                force = self.robot2.mass * (b2Vec2(vx2, vy2) - self.robot2.linearVelocity) * 30.0
                self.robot2.ApplyTorque(torque, wake=True)
                self.robot2.ApplyForce(force, self.robot2.position, wake=True)
            self.world.Step(self.time_step, self.vel_iters, self.pos_iters)

        # calculate costs
        cost1 = np.linalg.norm(np.array(self.goal1) - np.array(self.obj1.position))
        cost2 = np.linalg.norm(np.array(self.goal2) - np.array(self.obj2.position))
        y = cost1 + cost2 - self.initial_dist 
        return y  # type: ignore
