from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))  # path handling for supercloud

from pexps.exp import *
from pexps.env import *
from u import *


class RingEnv(Env):
    def def_sumo(self):
        self.fname = self.config.get("fname")

        self.rl_speeds = []
        self.actions = []
        self.lane_changes = self.config.get("lane_changes")
        self.num_attempts = 0
        self.all_speeds = []
        self.traj = {}
        r = self.config.circumference / (2 * np.pi)

        sumo_args = {
            "net": self.config.get("map_path"),
            "additional": self.config.get("route_path"),
            "emission": self.config.res / "emissions.xml",
        }
        return super().def_sumo({}, file_args=sumo_args)

    @property
    def stats(self):
        stats = {k: v for k, v in super().stats.items() if "flow" not in k}
        stats["circumference"] = self.config.circumference
        return stats

    def step(self, action=None, pcp_action=np.array(0), inferred_trait=None):
        config = self.config
        ts = self.ts
        max_speed = config.max_speed
        max_dist = config.circumference_max
        rl_type = ts.types.rl
        carla_type = ts.types["vehicle.audi.a2"]

        control_vehicles = rl_type
        if len(carla_type.vehicles) > 0:
            control_vehicles = carla_type

        # if there are no vehicles to control just step up the sim
        if not control_vehicles.vehicles:
            super().step()
            return config.observation_space.low, 0, False, 0

        rl = nexti(control_vehicles.vehicles)
        if action is not None:  # action is None only right after reset
            ts.tc.vehicle.setMinGap(
                rl.id, 0.1
            )  # Set the minGap to 0.1 after the warmup period so the vehicle doesn't crash during warmup
            vel = action
            if isinstance(vel, np.ndarray):
                vel = vel.item()

            # perform the action
            perform_action = vel + np.random.normal(self.driver_trait, 1)
            if config.get("aclip", True):
                perform_action = np.clip(perform_action, 0, max_speed)

            if not config.carla:
                ts.tc.vehicle.slowDown(rl.id, perform_action, 1e-3)
                config.log(
                    f"Performing action: {perform_action}; PCP action is {pcp_action}; PerP action is {vel}"
                )
            else:
                config.log(f"PCP action is {pcp_action}; PerP action is {vel}")
            self.actions.append(perform_action)

        rl_speed = [
            veh.speed
            for veh in (
                self.ts.types["vehicle.audi.a2"]
                if config.carla
                else self.ts.types.rl.vehicles
            )
        ]
        all_speed = [veh.speed for veh in self.ts.vehicles.values()]
        avg_speed = sum(all_speed) / len(all_speed)
        self.rl_speeds.append(rl_speed[0])
        self.all_speeds.append(avg_speed)

        super().step()

        if len(ts.new_arrived | ts.new_collided):
            print("Detected collision")
            return config.observation_space.low, -config.collision_penalty, True, None
        elif len(ts.vehicles) < config.n_veh:
            print("Bad initialization occurred, fix the initialization function")
            return config.observation_space.low, 0, True, None

        if inferred_trait is None:
            inferred_trait = (
                np.zeros(shape=(2)) if config.vae_trait_inference else np.array([0])
            )

        leader, dist = rl.leader()
        obs = [
            rl.speed / max_speed,
            leader.speed / max_speed,
            dist / max_dist,
            pcp_action.tolist() / max_speed,
        ]
        obs = np.clip(obs, 0, 1) * (1 - config.pcp_config.low) + config.pcp_config.low
        obs = np.array(obs.tolist() + inferred_trait.tolist())

        # We want to maximize the average speed while minimizing the speed error between the PCP and the current policy
        reward = (
            # Max Speed for all cars
            config.beta_speed
            * np.mean(
                [
                    v.speed
                    for v in (
                        ts.vehicles
                        if config.global_reward
                        else control_vehicles.vehicles
                    )
                ]
            )
            / config.max_speed
        )
        # Minimize difference in optimal speeds
        +(-1 * config.beta_error * abs(rl.speed - pcp_action.item()))

        self.last_speed = rl.speed
        return obs.astype(np.float32), reward, False, None


class Ring(Main):
    def create_env(self):
        return NormEnv(self, RingEnv(self))

    @property
    def observation_space(self):
        # observations are vector in R^(n_obs) constrained between 0 and 1 for all dims
        if self.perp:
            return Box(
                low=0,
                high=1,
                shape=(self._n_obs + (2 if self.vae_trait_inference else 1) + 1,),
                dtype=np.float32,
            )
        else:
            # For PCP
            return Box(low=0, high=1, shape=(self._n_obs,), dtype=np.float32)

    @property
    def action_space(self):
        if self.perp:
            return Box(
                low=-1 * self.offset_lim,
                high=self.offset_lim,
                shape=(1,),
                dtype=np.float32,
            )
        else:
            # For PCP
            return Discrete(self.n_actions)


if __name__ == "__main__":
    config = Ring.from_args(globals(), locals()).setdefaults(
        n_lanes=1,
        horizon=3000,
        warmup_steps=1000,
        sim_step=0.1,
        av=1,
        max_speed=35,
        max_accel=0.5,
        max_decel=0.5,
        circumference=640,
        circumference_max=654,
        circumference_min=628,
        initial_space="free",
        sigma=0.2,
        lc_av=False,
        act_type="vel_discrete",
        lc_act_type="continuous",
        low=-1,
        global_reward=False,
        accel_penalty=0,
        collision_penalty=100,
        rl_sigma=0.0,
        constant_rl_velocity=None,  # speed level, not absolute speed
        meta_stable_penalty=0,
        incorrect_penalty=0,
        hc_param=1,
        hc_reward="last",
        act_coef=1,
        n_steps=100,
        gamma=0.999,
        alg=PG,
        norm_reward=True,
        center_reward=True,
        adv_norm=False,
        step_save=5,
        n_actions=10,
        render=False,
        carla=False,
        redef_sumo=True,
        map_path="../saved/map_files/ring.net.xml",
        route_path="../saved/map_files/ring.route.xml",
        trait_inference_path="./saved/trait_2.5",
        vae_trait_inference=False,
        simple_trait_inference=True,
        pcp_path="../saved/pcp_models/hc_20",
        beta_speed=1,
        beta_error=1,
        perp=True,
        inference_len=20,
        offset_lim=6,
        aclip=True,
        _n_obs=3,
        lr=0.01,
    )
    config.trait_inference_path = Path(config.trait_inference_path)
    config.pcp_path = Path(config.pcp_path)

    # load pcp config
    pcp_config = Ring(Path(config.pcp_path)).setdefaults(
        use_critic=True,
        perp=False,
        _n_obs=3,
    )
    pcp_config.res = Path(config.pcp_path)
    pcp_config = pcp_config.load()
    config.pcp_config = pcp_config

    # PeRP and PCP need to have the same delta
    config.hc_param = config.pcp_config.hc_param

    if config.vae_trait_inference:
        # load Trait VAE config
        config.trait_inf_config = get_vae_config(
            Path(config.trait_inference_path) / "args.json"
        )

    config.setdefaults(n_veh=40, _n_obs=3)
    config.step_save = config.step_save or min(5, config.n_steps // 10)
    config.run()
