from u.exp import Config

from pexps.ut import *
from pexps.env import Env as FlowEnv

import socket
import json
import pdb
import glob, os


sys.path.append("../trait_estimation/")
from model import TraitVAE
from vae_utils import get_config as get_vae_config
import torch

class Main(Config):
    flow_base = "./FLOW_RES_DIR"

    never_save = {'trial', 'has_av', 'e', 'disable_amp', 'opt_level', 'tmp'} | Config.never_save


    def __init__(self, res, *args, **kwargs):
        tmp = Path(res)._real in [Path.env('HOME'), Main.flow_base]
        if tmp:
            res = Main.flow_base / 'tmp' / rand_string(8)
        kwargs.setdefault('disable_amp', True)
        kwargs.setdefault('tmp', tmp)
        super().__init__(res, *args, **kwargs)
        if tmp:
            os.environ['WANDB_MODE'] = 'dryrun'
        self.setdefaults(e=False, tb=True, wb=False)
        self.logger = self.logger and self.e is False
        if tmp and self.e is False:
            res.mk()
            self.log('Temporary run for testing with res=%s' % res)

    def create_env(self):
        raise NotImplementedError

    @property
    def dist_class(self):
        if '_dist_class' not in self:
            self._dist_class = build_dist(self.action_space)
        return self._dist_class

    @property
    def model_output_size(self):
        if '_model_output_size' not in self:
            self._model_output_size = self.dist_class.model_output_size
        return self._model_output_size

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    def set_model(self):
        self._model = self.get('model_cls', FFN)(self)
        return self

    def schedule(self, coef, schedule=None):
        if not schedule and isinstance(coef, (float, int)):
            return coef
        frac = self._i / self.n_steps
        frac_left = 1 - frac
        if callable(coef):
            return coef(frac_left)
        elif schedule == 'linear':
            return coef * frac_left
        elif schedule == 'cosine':
            return coef * (np.cos(frac * np.pi) + 1) / 2

    @property
    def _lr(self):
        return self.schedule(self.get('lr', 1e-4), self.get('lr_schedule'))

    def log_stats(self, stats, ii=None, n_ii=None, print_time=False):
        if ii is not None:
            assert n_ii is not None and n_ii > 0
        stats = {k: v for k, v in stats.items() if v is not None}
        total_time = time() - self._run_start_time
        if print_time:
            stats['total_time'] = total_time

        prints = []
        if ii is not None:
            prints.append('ii {:2d}'.format(ii))

        prints.extend('{} {:.3g}'.format(*kv) for kv in stats.items())

        widths = [len(x) for x in prints]
        line_w = terminal_width()
        prefix = 'i {:d}'.format(self._i)
        i_start = 0
        curr_w = len(prefix) + 3
        curr_prefix = prefix
        for i, w in enumerate(widths):
            if curr_w + w > line_w:
                self.log(' | '.join([curr_prefix, *prints[i_start: i]]))
                i_start = i
                curr_w = len(prefix) + 3
                curr_prefix = ' ' * len(prefix)
            curr_w += w + 3
        self.log(' | '.join([curr_prefix, *prints[i_start:]]))
        sys.stdout.flush()

        if n_ii is not None:
            self._writer_buffer.append(**stats)
            if ii == n_ii - 1:
                stats = {k: np.mean(self._writer_buffer.pop(k)) for k in stats}
            else:
                return

        df = self._results
        for k, v in stats.items():
            if k not in df:
                df[k] = np.nan
            df.loc[self._i, k] = v

        if self.e is False:
            if self.tb:
                for k, v in stats.items():
                    self._writer.add_scalar(k, v, global_step=self._i, walltime=total_time)
            if self.wb:
                self._writer.log(stats, step=self._i)

    def get_log_ii(self, ii, n_ii, print_time=False):
        return lambda **kwargs: self.log_stats(kwargs, ii, n_ii, print_time=print_time)

    def on_rollout_worker_start(self):
        self._env = self.create_env()
        self.use_critic = False # Don't need value function on workers
        self.set_model()
        self._model.eval()
        self._i = 0

    def set_weights(self, weights): # For Ray
        self._model.load_state_dict(weights, strict=False) # If c.use_critic, worker may not have critic weights

    def on_train_start(self):
        self.setdefaults(alg='Algorithm')
        self._env = self.create_env()

        self._alg = (eval(self.alg) if isinstance(self.alg, str) else self.alg)(self)
        self.set_model()
        self._model.train()
        self._model.to(self.device)

        self._i = 0 # for c._lr
        opt = self.get('opt', 'Adam')
        if opt == 'Adam':
            self._opt = optim.Adam(self._model.parameters(), lr=self._lr, betas=self.get('betas', (0.9, 0.999)), weight_decay=self.get('l2', 0))
        elif opt == 'RMSprop':
            self._opt = optim.RMSprop(self._model.parameters(), lr=self._lr, weight_decay=self.get('l2', 0))

        self._run_start_time = time()
        self._i = self.set_state(self._model, opt=self._opt, step='max')
        if self._i:
            self._results = self.load_train_results().loc[:self._i]
            self._run_start_time -= self._results.loc[self._i, 'total_time']
        else:
            self._results = pd.DataFrame(index=pd.Series(name='step'))
        self._i_gd = None

        self.try_save_commit(Main.flow_base)

        if self.tb:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.res, flush_secs=10)
        if self.wb:
            import wandb
            wandb_id_path = (self.res / 'wandb' / 'id.txt').dir_mk()
            self._wandb_run = wandb.init( # name and project should be set as env vars
                name=self.res.rel(Path.env('FA')),
                dir=self.res,
                id=wandb_id_path.load() if wandb_id_path.exists() else None,
                config={k: v for k, v in self.items() if not k.startswith('_')},
                save_code=False
            )
            wandb_id_path.save(self._wandb_run.id)
            self._writer = wandb
        self._writer_buffer = NamedArrays()

    def on_step_start(self, stats={}):
        lr = self._lr
        for g in self._opt.param_groups:
            g['lr'] = float(lr)
        self.log_stats(dict(**stats, **self._alg.on_step_start(), lr=lr))

        if self._i % self.step_save == 0:
            self.save_train_results(self._results)
            self.save_state(self._i, self.get_state(self._model, self._opt, self._i))

    def rollouts(self):
        """ Collect a list of rollouts for the training step """
        if self.use_ray:
            import ray
            weights_id = ray.put({k: v.cpu() for k, v in self._model.state_dict().items()})
            [w.set_weights.remote(weights_id) for w in self._rollout_workers]
            rollout_stats = flatten(ray.get([w.rollouts_single_process.remote() for w in self._rollout_workers]))
        else:
            rollout_stats = self.rollouts_single_process()
        rollouts = [self.on_rollout_end(*rollout_stat, ii=ii, n_ii=self.n_rollouts_per_step) for ii, rollout_stat in enumerate(rollout_stats)]
        return NamedArrays.concat(rollouts, fn=flatten)

    def rollouts_single_process(self):
        if self.n_rollouts_per_worker > 1:
            rollout_stats = [self.var(i_rollout=i).rollout() for i in range(self.n_rollouts_per_worker)]
        else:
            n_steps_total = 0
            rollout_stats = []
            while n_steps_total < self.horizon:
                rollout, stats = self.rollout()
                rollout_stats.append((rollout, stats))
                n_steps_total += stats.get('horizon') or len(stats['reward'])
        return rollout_stats

    def get_env_stats(self):
        return self._env.stats

    def rollout(self):
        self.setdefaults(skip_stat_steps=0, i_rollout=0, rollout_kwargs=None)
        if self.rollout_kwargs and self.e is False:
            self.update(self.rollout_kwargs[self.i_rollout])
        t_start = time()

        ret = self._env.reset()

        self.log(f"Driver Trait: {self._env.driver_trait}")

        if not isinstance(ret, dict):
            ret = dict(obs=ret)
        rollout = NamedArrays()
        rollout.append(**ret)

        done = False
        a_space = self.action_space
        step = 0
        actions = []
        
        while step < self.horizon + self.skip_stat_steps and not done:
            # This has to be here since it should only be queried every Delta times and not at every step
            # The hold length parameter dictates when the driver trait and as well as the PCP action.
            # Infer the driver trait for the hold length
            if self.vae_trait_inference:
                trait_input = np.array(rollout.obs[-self.inference_len:])[:, :3].reshape(1, -1, 3).astype(np.float32)
                inferred_trait = from_torch(self.traitVAE(to_torch(trait_input)))[0].flatten()
            elif self.simple_trait_inference:
                inferred_trait = self._env.driver_trait
            else:
                inferred_trait = np.array([0])

            # get the PCP action 
            pcp_pred = from_torch(self.pcp_model(to_torch(rollout.obs[-1][:3]), value=False, policy=True, argmax=False))
            pcp_pred.action = np.array(pcp_pred.action.item() / (self.pcp_config.n_actions - 1) * self.pcp_config.max_speed)
                
            if self.perp:
                # get the PeRP action
                pred = from_torch(self._model(to_torch(rollout.obs[-1]), value=False, policy=True, argmax=False))
                if self.get('aclip', True) and isinstance(a_space, Box):
                    pred.action = np.clip(pred.action, a_space.low, a_space.high)

                pred.action = pcp_pred.action + pred.action
            else:
                pred = pcp_pred
            
            # Send action to Carla Client
            if self.carla and self.client_conn:
                self.client_conn.send(f"{pred.action.item()};".encode())

            r = 0.
            ## MAYURI UPDATE
            rollout.append(**pred)
            for i in range(self.hc_param):
                # self.log(f"{i}: {rollout.action[-1]}, {pcp_pred.action}, {inferred_trait}")
                ret_inner = self._env.step(rollout.action[-1], pcp_pred.action, inferred_trait)

                if isinstance(ret_inner, tuple):
                    obs, reward, done, info = ret_inner
                    ret_inner = dict(obs=obs, reward=reward, done=done, info=info)
                    if i == 0:  # first step corresponds to start of a semi-step
                        ret = dict(obs=obs, reward=reward, done=done, info=info)
                elif i == 0:
                    ret = copy(ret_inner)
                # done = ret.setdefault('done', False)

                r += ret_inner['reward']
                actions.append(pred.action)

                if done:
                    ret['done'] = done
                    ret['reward'] = reward
                    ret = {k: v for k, v in ret.items() if k not in ['obs', 'id']}
                    break  # to ensure episode termination for hc_param > 1

            # default: c.hc_reward == 'last'
            if self.get('hc_reward') and self.hc_reward == 'average':
                r /= self.hc_param
                ret['reward'] = r

            rollout.append(**ret)
            step += self.hc_param

        self._env.actions = actions
        stats = dict(rollout_time=time() - t_start, **self.get_env_stats())
        return rollout, stats

    def on_rollout_end(self, rollout, stats, ii=None, n_ii=None):
        """ Compute value, calculate advantage, log stats """
        t_start = time()
        step_id_ = rollout.pop('id', None)
        done = rollout.pop('done', None)
        multi_agent = step_id_ is not None

        step_obs_ = rollout.obs
        step_obs = step_obs_ if done[-1] else step_obs_[:-1]
        assert len(step_obs) == len(rollout.reward)

        value_ = None
        if self.use_critic:
            (_, mb_), = rollout.filter('obs').iter_minibatch(concat=multi_agent, device=self.device)
            value_ = from_torch(self._model(mb_.obs, value=True).value.view(-1))

        if multi_agent:
            step_n = [len(x) for x in rollout.reward]
            reward = np.concatenate(rollout.reward)
            ret, adv = calc_adv_multi_agent(np.concatenate(step_id_), reward, c.gamma, value_=value_, lam=self.lam)
            rollout.update(obs=step_obs, ret=split(ret, step_n))
            if self.use_critic:
                rollout.update(value=split(value_[:len(ret)], step_n), adv=split(adv, step_n))
        else:
            reward = rollout.reward
            ret, adv = calc_adv(reward, self.gamma, value_, self.lam)
            rollout.update(obs=step_obs, ret=ret)
            if self.use_critic:
                rollout.update(value=value_[:len(ret)], adv=adv)

        log = self.get_log_ii(ii, n_ii)
        log(**stats)
        log(
            reward_mean=np.mean(reward),
            value_mean=np.mean(value_) if self.use_critic else None,
            ret_mean=np.mean(ret),
            adv_mean=np.mean(adv) if self.use_critic else None,
            explained_variance=explained_variance(value_[:len(ret)], ret) if self.use_critic else None
        )
        log(rollout_end_time=time() - t_start)
        return rollout

    def on_step_end(self, stats={}):
        self.log_stats(stats, print_time=True)
        self.log('')

    def on_train_end(self):
        if self._results is not None:
            self.save_train_results(self._results)

        save_path = self.save_state(self._i, self.get_state(self._model, self._opt, self._i))
        if self.tb:
            self._writer.close()
        if hasattr(self._env, 'close'):
            self._env.close()

        # close connection to carla
        if self.client_conn:
            self.client_conn.close()

    def train(self):
        self.on_train_start()
        while self._i < self.n_steps:
            self.on_step_start()
            with torch.no_grad():
                rollouts = self.rollouts()
            gd_stats = {}
            if len(rollouts.obs):
                t_start = time()
                self._alg.optimize(rollouts)
                gd_stats.update(gd_time=time() - t_start)
            self.on_step_end(gd_stats)
            self._i += 1
        self.on_step_start() # last step
        
        with torch.no_grad():
            rollouts = self.rollouts()
            self.on_step_end()
        self.on_train_end()

    def eval(self):
        self.setdefaults(alg='PPO')
        
        self._env = self.create_env()

        # SUMO needs to be generated before this can take place
        # Connect to the carla client
        if self.carla:
            # This is blocking and will wait until a connection is established
            self.client_conn = self.setup_server() 

        self._alg = (eval(self.alg) if isinstance(self.alg, str) else self.alg)(self)
        self.set_model()
        self._model.eval().to(self.device)
        self._results = pd.DataFrame(index=pd.Series(name='step'))
        self._writer_buffer = NamedArrays()

        kwargs = {'step' if isinstance(self.e, int) else 'path': self.e}
        step = self.set_state(self._model, opt=None, **kwargs)
        self.log('Loaded model from step %s' % step)
        
        self._run_start_time = time()
        self._i = 1
        for _ in range(self.n_steps):
            self.rollouts()
            if self.get('result_save'):
                self._results.to_csv(self.result_save)
            if self.get('vehicle_info_save'):
                self._env.vehicle_info.to_csv(self.vehicle_info_save)
                self._env.sumo_paths['net'].cp(self.vehicle_info_save.replace('.csv', '.net.xml'))
            ## MAYURI UPDATE
            rl_actions = self._env.actions
            rl_speed = self._env.rl_speeds
            avg_speed = self._env.all_speeds
            lc = self._env.lane_changes
            hc = self.hc_param
            e = self.e
            import pickle
            if self.get('fname'):
                name = 'evals/' + self.fname + '_hc={}_g={}_e={}_t={}'.format(self.hc_param, self.global_reward, self.e, b)
                f = open(name + '_rl_speed', 'wb')
                pickle.dump(rl_speed, f)
                f.close()
                f = open(name + '_rl_actions', 'wb')
                pickle.dump(rl_actions, f)
                f.close()
                f = open(name + '_all_speed', 'wb')
                pickle.dump(avg_speed, f)
                f.close()
                f = open(name + '_traj', 'wb')
                pickle.dump(self._env.traj, f)
                f.close()

            self._i += 1
            self.log('')
        if hasattr(self._env, 'close'):
            self._env.close()
        
        # close connection to carla
        if self.client_conn:
            self.client_conn.close()
        
    def setup_server(self):
        port = 5000

        server_socket = socket.socket()
        server_socket.bind(('', port))

        print(f"Listening for client connections at port {port}")
        server_socket.listen(1) # arg is the number of clients that the server can listen to simultaneously
        conn, address = server_socket.accept() # this is blocking
        print(f"Connected to {address}")

        conn.send("Connected".encode())
        return conn

    def run(self):
        import re
        numbers = re.compile(r'(\d+)')
        def numericalSort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
        
        self.log(format_yaml({k: v for k, v in self.items() if not k.startswith('_')}))
        self.setdefaults(n_rollouts_per_step=1)

        self.client_conn = None
        
        # load the Piecewise constant policy
        pcp_model = self.pcp_config.get('model_cls', FFN)(self.pcp_config).to(self.device)
        best_model_path = max(glob.glob(self.pcp_path / "models/*"), key=numericalSort)
        pcp_model.load_state_dict(torch.load(best_model_path, map_location=self.device)['net'])
        self.pcp_model = pcp_model.eval()
        self.log(f"loaded PCP from {best_model_path}")
        
        if self.vae_trait_inference:
            # load the trait inference model
            model = TraitVAE(self.trait_inf_config).to(self.device)
            model.load_state_dict(torch.load(self.trait_inference_path / "model_TraitVAE.pt", map_location=self.device)['state_dict'])
            self.traitVAE = model.eval()
            self.log(f"loaded trait VAE from {self.trait_inference_path}")

        if self.e:
            self.n_workers = 1
            self.setdefaults(use_ray=False, n_rollouts_per_worker=self.n_rollouts_per_step // self.n_workers)
            self.eval()
        else:
            self.setdefaults(device='cuda' if torch.cuda.is_available() else 'cpu')
            if self.get('use_ray', True) and self.n_rollouts_per_step > 1 and self.get('n_workers', np.inf) > 1:
                self.setdefaults(n_workers=self.n_rollouts_per_step, use_ray=True)
                self.n_rollouts_per_worker = self.n_rollouts_per_step // self.n_workers
                import ray
                try:
                    ray.init(num_cpus=self.n_workers, include_dashboard=False)
                except:
                    ray.init(num_cpus=self.n_workers, include_dashboard=False, _temp_dir=(Path.env('FLOW_RES_DIR') / 'tmp')._real)
                RemoteMain = ray.remote(type(self))
                worker_kwargs = self.get('worker_kwargs') or [{}] * self.n_workers
                assert len(worker_kwargs) == self.n_workers
                worker_kwargs = [{**self, 'main': False, 'device': 'cpu', **args} for args in worker_kwargs]
                self._rollout_workers = [RemoteMain.remote(**kwargs, i_worker=i) for i, kwargs in enumerate(worker_kwargs)]
                ray.get([w.on_rollout_worker_start.remote() for w in self._rollout_workers])
            else:
                self.setdefaults(n_workers=1, n_rollouts_per_worker=self.n_rollouts_per_step, use_ray=False)
            assert self.n_workers * self.n_rollouts_per_worker == self.n_rollouts_per_step
            self.train()
