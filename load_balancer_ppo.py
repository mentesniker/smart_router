import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import threading
import time
import queue as queue_module
import random

class Task:
    """A task with per-server processing durations and CPU utilizations."""

    def __init__(self):
        # Server 1 is most powerful (fastest, least CPU)
        self.durations = [
            random.uniform(0.05, 0.15),  # Server 1 (fastest)
            random.uniform(0.10, 0.20),  # Server 2
            random.uniform(0.30, 0.40),  # Server 3 (slowest)
        ]
        self.cpu_utilizations = [
            random.uniform(0.2, 0.4),  # Server 1 (least CPU)
            random.uniform(0.3, 0.5),  # Server 2
            random.uniform(0.6, 0.7),  # Server 3 (most CPU)
        ]


class Server:
    """Server that processes tasks asynchronously from its waiting queue."""

    def __init__(self, server_id, max_queue_size, num_threads=4):
        self.server_id = server_id
        self.cpu_utilization = 0.0
        self.task_queue = queue_module.Queue(maxsize=max_queue_size)
        self.max_queue_size = max_queue_size
        self.running = False
        self.lock = threading.Lock()
        self.num_threads = num_threads
        self.threads = []

    def serve(self):
        """Infinite loop that processes tasks from the queue."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.01)
                duration = task.durations[self.server_id]
                cpu_cost = task.cpu_utilizations[self.server_id]
                with self.lock:
                    self.cpu_utilization += cpu_cost
                time.sleep(duration)
                with self.lock:
                    self.cpu_utilization = max(0.0, self.cpu_utilization - cpu_cost)
                self.task_queue.task_done()
            except queue_module.Empty:
                continue

    def start(self):
        self.running = True
        self.threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.serve, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=0.1)

    def reset(self):
        """Stop processing, clear the queue, refill with a random number of tasks, and restart."""
        self.stop()
        # Drain existing queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue_module.Empty:
                break
        # Reset CPU utilization
        with self.lock:
            self.cpu_utilization = 0.0
        # Refill with a random number of tasks (up to max capacity)
        num_tasks = random.randint(0, self.max_queue_size)
        for _ in range(num_tasks):
            try:
                self.task_queue.put_nowait(Task())
            except queue_module.Full:
                break
        # Restart worker threads
        self.start()

    @property
    def queue_length(self):
        return self.task_queue.qsize()

    def enqueue(self, task):
        """Enqueue a task. Returns True if successful, False if queue is full."""
        try:
            self.task_queue.put_nowait(task)
            return True
        except queue_module.Full:
            return False


class Actor(nn.Module):
    """Router neural network: state -> probability distribution over servers."""

    def __init__(self, state_dim, num_servers, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_servers),
        )

    def forward(self, state):
        return torch.softmax(self.net(state), dim=-1)

    def get_logits(self, state):
        return self.net(state)


class LyapunovCritic:
    """MINLP critic that solves for optimal server assignment using cvxpy."""

    def __init__(self, num_servers, max_queue_sizes, c=5.0, d=6.0):
        self.num_servers = num_servers
        self.max_queue_sizes = max_queue_sizes
        self.c = c
        self.d = d
        self.virtual_queue = 0
        self.weights = [1, 1, 1]
        self.actual_queue_lengths_after_action = [0,0,0]

    def solve(self, server_queue_lengths, cpu_utils, current_lost_tasks=None, ratio=0):
        """
        Solve MINLP: min |u + d|^2
        s.t. sum(u) == 1, u binary (one-hot: choose one server)
            ΔH(Q_i) - g(H(Q_i)) <= 0.8 * max_queue_i  for each server i
            ΔV(Q) <= -c * P(Q)
            ΔL(t) - g(L(t)) <= threshold  (lost tasks drift constraint)

        where:
            ΔH(Q_i) = Q_i(t+1) - Q_i(t)  (queue drift per server)
            g(Q_i) = Q_i(t)^2              (quadratic penalty)
            V(Q) = sum(0.5 * Q_i^2)        (Lyapunov function)
            P(Q) = total CPU utilization
            ΔL(t) = L(t+1) - L(t)          (lost tasks drift)
            L(t) = current_lost_tasks      (cumulative lost tasks)
        """
        if(np.sum(server_queue_lengths)==0):
            return torch.tensor([1,0,0])
        preference = np.array([0.01,0.02,0.03])  # lower = preferred
        u = cp.Variable(self.num_servers, boolean=True)

        # Slack variables for soft constraints
        s_q_1 = cp.Variable(nonneg=True)  # slack for queue capacity constraints
        s_q_2 = cp.Variable(nonneg=True)  # slack for queue capacity constraints
        s_q_3 = cp.Variable(nonneg=True)  # slack for queue capacity constraints
        s_q = cp.Variable(self.num_servers,nonneg=True)  # slack for queue capacity constraints
        s_v = cp.Variable(nonneg=True)                    # slack for Lyapunov drift constraint

        queue_capacity_constraints = [s_q_1,s_q_2,s_q_3]
        slack_penalty_v = 10
        #slack_penalty_q = 1e3
        slack_penalty_q = np.array([1.0, 50, 2500])
        slack_penalty_q_1 = 1
        slack_penalty_q_2 = 1
        slack_penalty_q_3 = 8
        objective = cp.Minimize(
            cp.sum_squares(preference @ u
            + slack_penalty_q_1 * s_q_1
            + slack_penalty_q_2 * s_q_2
            + slack_penalty_q_3 * s_q_3
            #+ slack_penalty_q @ s_q
            + slack_penalty_v * s_v)
        )
        constraints = [cp.sum(u) == 1]

        # Per-server queue capacity constraints: ΔH(Q_i) + g(Q_i) <= 0.9 * max_queue_i + s_q[i]
        for i in range(self.num_servers):
            q_i = float(server_queue_lengths[i])
            q_base_i = float(self.actual_queue_lengths_after_action[i])
            delta_h = (q_i + u[i]) - q_base_i
            g_q = q_i
            #constraints.append(self.weights[i]*(delta_h + g_q) <= 0.9 * self.max_queue_sizes[i] + queue_capacity_constraints[i])
            constraints.append(self.weights[i]*(delta_h + g_q) <= (0.9+(ratio/10)) * self.max_queue_sizes[i] + queue_capacity_constraints[i])

        # Lyapunov drift constraint: ΔV(Q) <= -c * P(Q) + s_v
        if self.actual_queue_lengths_after_action is not None:
            v_current = sum(0.5 * q for q in self.actual_queue_lengths_after_action)
        else:
            v_current = sum(0.5 * q for q in server_queue_lengths)
        v_next_terms = []
        for i in range(self.num_servers):
            q_i = float(server_queue_lengths[i])
            q_base_i = abs(q_i - float(self.actual_queue_lengths_after_action[i]))
            v_next_terms.append(0.5 * (q_base_i + u[i])**2)
        delta_v = (sum(v_next_terms)) - (v_current)
        total_cpu = sum(float(c) for c in cpu_utils)
        constraints.append(delta_v <= -1 * self.c * total_cpu + s_v)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCIP, verbose=False)
        except cp.SolverError:
            print("FAILURE ALERT!!! CRITICAL PROBLEM!!!")
            print(server_queue_lengths)
            return self._smart_fallback(server_queue_lengths)

        if problem.status in ("infeasible", "unbounded") or u.value is None:
            print("FAILURE ALERT!!! CRITICAL " + str(problem.status) +"!!!")
            print(server_queue_lengths)
            #fallback = np.zeros(self.num_servers)
            #fallback[0] = 1.0
            return self._smart_fallback(server_queue_lengths)
            #return fallback

        one_hot = np.zeros(self.num_servers)
        one_hot[int(np.argmax(u.value))] = 1.0
        return one_hot
    
    def _smart_fallback(self, server_queue_lengths):
        """Fallback: pick the server with the most remaining capacity."""
        ratios = [
            server_queue_lengths[i] / self.max_queue_sizes[i]
            for i in range(self.num_servers)
        ]
        best = int(np.argmin(ratios))
        one_hot = np.zeros(self.num_servers)
        one_hot[best] = 1.0
        return one_hot

    def update_actual_queues(self, actual_queue_lengths):
        """Store actual queue lengths observed after taking an action."""
        self.actual_queue_lengths_after_action = list(actual_queue_lengths)


class PPOLoadBalancer:
    """PPO-based load balancer with threaded servers and Lyapunov MINLP critic."""

    def __init__(
        self,
        max_queue_sizes=None,
        router_queue_size=100,
        arrival_rate=1.0,
        c=5.0,
        d=6.0,
        lr=1e-3,
        num_episodes=10,
        steps_per_episode=1000,
    ):
        self.num_servers = 3
        self.max_queue_sizes = max_queue_sizes or [20, 10, 5]
        self.router_queue_size = router_queue_size
        self.arrival_rate = arrival_rate
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode

        # State: cpu_utils(3) + router_queue(1) + server_queues(3) = 7
        self.state_dim = 2 * self.num_servers + 1

        self.servers = [
            Server(i, self.max_queue_sizes[i], 4-(i+1)) for i in range(self.num_servers)
        ]
        self.router_queue = queue_module.Queue(maxsize=router_queue_size)
        self.actor = Actor(self.state_dim, self.num_servers)
        self.critic = LyapunovCritic(self.num_servers, self.max_queue_sizes, c, d)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # PPO hyperparameters
        self.ppo_clip_eps = 0.2
        self.ppo_epochs = 4
        self.ppo_batch_size = 64
        self.entropy_coeff = 0.01
        self.max_grad_norm = 0.5

        # Logging (per-step)
        self.queue_history = [[] for _ in range(self.num_servers)]
        self.cpu_history = []
        self.tasks_lost_history = []
        self.chosen_server_history = []
        self.total_tasks_lost = 0
        self.total_tasks_routed = 0
        self.lost_ratio_history = []
        self.loss_history = []
        self._running = False
        self._arrival_sleep_range = (0.05, 0.10)

    def get_state(self):
        """Read current system state (thread-safe)."""
        cpu_utils = []
        for s in self.servers:
            with s.lock:
                cpu_utils.append(s.cpu_utilization)
        queue_lengths = [s.queue_length for s in self.servers]
        router_q = self.router_queue.qsize()

        state = np.array(
            [min(c, 1.0) for c in cpu_utils]
            + [router_q / self.router_queue_size]
            + [queue_lengths[i] / self.max_queue_sizes[i] for i in range(self.num_servers)],
            dtype=np.float32,
        )
        return state, cpu_utils, queue_lengths, router_q

    def _arrival_loop(self):
        """Separate thread: generate Poisson arrivals into the router queue."""
        while self._running:
            num_arrivals = np.random.poisson(self.arrival_rate)
            for _ in range(num_arrivals):
                task = Task()
                try:
                    self.router_queue.put_nowait(task)
                except queue_module.Full:
                    self.total_tasks_lost += 1
            lo, hi = self._arrival_sleep_range
            time.sleep(random.uniform(lo, hi))

    def _ppo_update(self, states, actions, old_log_probs, rewards):
        """PPO clipped surrogate objective update over collected trajectory."""
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_log_probs_t = torch.FloatTensor(old_log_probs).detach()

        # Normalize rewards for stability
        rewards_arr = np.array(rewards, dtype=np.float32)
        if rewards_arr.std() > 1e-8:
            rewards_arr = (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)
        rewards_t = torch.FloatTensor(rewards_arr)

        total_loss = 0.0
        n = len(states)
        for _ in range(self.ppo_epochs):
            # Shuffle indices for mini-batch updates
            indices = np.random.permutation(n)
            for start in range(0, n, self.ppo_batch_size):
                end = min(start + self.ppo_batch_size, n)
                idx = indices[start:end]

                batch_states = states_t[idx]
                batch_actions = actions_t[idx]
                batch_old_log_probs = old_log_probs_t[idx]
                batch_rewards = rewards_t[idx]

                # Current policy log-probs and entropy
                logits = self.actor.get_logits(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps)
                surrogate = torch.min(ratio * batch_rewards, clipped_ratio * batch_rewards)
                loss = -surrogate.mean() - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss

    def _router_loop(self):
        """Separate thread: router pulls tasks, decides routing, and trains actor."""
        for episode in range(self.num_episodes):
            # Randomize arrival sleep interval for this epoch
            lo = random.uniform(0.01, 0.08)
            hi = lo + random.uniform(0.02, 0.10)
            self._arrival_sleep_range = (lo, hi)

            episode_loss = 0.0
            episode_routed = 0
            this_step_lost_tasks = 0
            this_step_total_per_server = [0, 0, 0]
            current_lost_tasks = 0

            # Trajectory buffer for PPO
            buf_states = []
            buf_actions = []
            buf_log_probs = []
            buf_rewards = []

            this_epoch_total_routed_tasks = 0

            ratio = 0

            for step in range(self.steps_per_episode):
                state_arr, cpu_utils, q_lens, router_q = self.get_state()

                # Log current state
                for i in range(self.num_servers):
                    self.queue_history[i].append(q_lens[i])
                self.cpu_history.append(sum(cpu_utils))
                self.tasks_lost_history.append(self.total_tasks_lost)

                # Try to route one task from the router queue
                try:
                    task = self.router_queue.get_nowait()
                    this_epoch_total_routed_tasks += 1
                except queue_module.Empty:
                    self.chosen_server_history.append(-1)
                    time.sleep(0.002)
                    continue

                # Sample action from actor policy
                state_tensor = torch.FloatTensor([state_arr])
                with torch.no_grad():
                    logits = self.actor.get_logits(state_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                chosen_server = action.item()

                # Critic solves MINLP for optimal server assignment
                target_one_hot = self.critic.solve(q_lens, cpu_utils, current_lost_tasks,ratio)
                target_server = int(np.argmax(target_one_hot))

                #print(target_server)

                # Route task to the critic's chosen server (expert action)
                success = self.servers[target_server].enqueue(task)
                if not success:
                    self.total_tasks_lost += 1
                    this_step_lost_tasks += 1
                    current_lost_tasks += 1
                self.total_tasks_routed += 1
                ratio = this_step_lost_tasks / this_epoch_total_routed_tasks
                self.lost_ratio_history.append(ratio)
                self.chosen_server_history.append(target_server)

                # Store actual queue lengths after action for drift calculation
                actual_q_lens = [s.queue_length for s in self.servers]
                self.critic.update_actual_queues(actual_q_lens)
                this_step_total_per_server[target_server] += 1
                episode_routed += 1

                # Reward: +1 if actor agrees with critic, -1 otherwise
                #reward = 1.0 if chosen_server == target_server else -1.0
                reward = torch.sum((log_prob - target_one_hot) ** 2)

                # Store transition in trajectory buffer
                buf_states.append(state_arr)
                buf_actions.append(chosen_server)
                buf_log_probs.append(log_prob.item())
                buf_rewards.append(reward)

                # PPO update every ppo_batch_size steps
                if len(buf_states) >= self.ppo_batch_size:
                    batch_loss = self._ppo_update(buf_states, buf_actions, buf_log_probs, buf_rewards)
                    episode_loss += batch_loss
                    self.loss_history.append(batch_loss)
                    buf_states.clear()
                    buf_actions.clear()
                    buf_log_probs.clear()
                    buf_rewards.clear()

            # Flush remaining transitions
            if buf_states:
                batch_loss = self._ppo_update(buf_states, buf_actions, buf_log_probs, buf_rewards)
                episode_loss += batch_loss
                self.loss_history.append(batch_loss)

            # Episode summary
            queue_str = ", ".join(
                f"S{i+1}={q}" for i, q in enumerate(this_step_total_per_server)
            )
            print(
                f"Episode {episode+1}/{self.num_episodes} | "
                f"Avg Queues: [{queue_str}] | "
                f"Tasks Lost: {this_step_lost_tasks} | "
                f"Total: {this_epoch_total_routed_tasks} | "
                f"Loss: {episode_loss:.4f}"
            )

            print(ratio)

            for server in self.servers:
                server.reset()

        self._running = False

    def train(self):
        """Start all threads and run training."""
        self._running = True

        # Start server threads
        for s in self.servers:
            s.start()

        # Start arrival generator thread
        arrival_thread = threading.Thread(target=self._arrival_loop, daemon=True)
        arrival_thread.start()

        # Start router thread (training happens here)
        router_thread = threading.Thread(target=self._router_loop, daemon=True)
        router_thread.start()

        # Wait for training to complete
        router_thread.join()

        # Stop servers
        for s in self.servers:
            s.stop()

    def plot_results(self):
        """Plot a 3x3 grid of training metrics."""
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()
        colors = ["tab:blue", "tab:green", "tab:purple"]

        # 1-3: Server queue dynamics
        for i in range(self.num_servers):
            axes[i].plot(self.queue_history[i], linewidth=0.8, color=colors[i])
            axes[i].axhline(
                y=self.max_queue_sizes[i], color="red", linestyle="--",
                linewidth=1.0, label=f"Max ({self.max_queue_sizes[i]})",
            )
            axes[i].axhline(
                y=self.max_queue_sizes[i]*0.9, color="red", linestyle="--",
                linewidth=1.0, label=f"Constraint ({self.max_queue_sizes[i]*0.9})",
            )
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel("Queue Length")
            axes[i].set_title(f"Server {i+1} Queue Dynamics")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        # 4: Total tasks lost (cumulative)
        axes[3].plot(self.tasks_lost_history, linewidth=0.8, color="tab:red")
        axes[3].set_xlabel("Step")
        axes[3].set_ylabel("Cumulative Tasks Lost")
        axes[3].set_title("Total Tasks Lost")
        axes[3].grid(True, alpha=0.3)

        # 5: Total CPU utilization per step
        axes[4].plot(self.cpu_history, linewidth=0.8, color="tab:orange")
        axes[4].set_xlabel("Step")
        axes[4].set_ylabel("Total CPU Utilization")
        axes[4].set_title("Total CPU Utilization per Step")
        axes[4].grid(True, alpha=0.3)

        # 6: Chosen server per step
        valid = [(i, s) for i, s in enumerate(self.chosen_server_history) if s >= 0]
        if valid:
            xs, ys = zip(*valid)
            axes[5].scatter(xs, ys, s=1, alpha=0.3, color="tab:cyan")
        axes[5].set_xlabel("Step")
        axes[5].set_ylabel("Chosen Server")
        axes[5].set_title("Chosen Server per Step")
        axes[5].set_yticks([0, 1, 2])
        axes[5].set_yticklabels(["Server 1", "Server 2", "Server 3"])
        axes[5].grid(True, alpha=0.3)

        # 7: Lost tasks ratio (lost / total)
        if self.lost_ratio_history:
            axes[6].plot(self.lost_ratio_history, linewidth=0.8, color="tab:brown")
            axes[6].axhline(
                y=0.20, color="red", linestyle="--",
                linewidth=1.0, label="Threshold (0.20)",
            )
            axes[6].set_xlabel("Step")
            axes[6].set_ylabel("Lost / Total Ratio")
            axes[6].set_title("Lost Tasks Ratio vs Steps")
            axes[6].legend()
            axes[6].grid(True, alpha=0.3)

        # 8: Loss per routing step
        if self.loss_history:
            axes[7].plot(self.loss_history, linewidth=0.8, color="tab:olive")
            axes[7].set_xlabel("Routing Step")
            axes[7].set_ylabel("Loss")
            axes[7].set_title("Actor Loss vs Iterations")
            axes[7].grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(8, 9):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig("load_balancer_results.png", dpi=150)
        plt.show()
        print("Plot saved to load_balancer_results.png")


if __name__ == "__main__":
    ppo = PPOLoadBalancer(
        max_queue_sizes=[60, 30, 14],
        router_queue_size=10000000,
        arrival_rate=128.0,
        c=3.0,
        d=4.0,
        lr=1e-3,
        num_episodes=4,
        steps_per_episode=1000,
    )
    ppo.train()
    ppo.plot_results()
