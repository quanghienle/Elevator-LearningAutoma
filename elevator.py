import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate


def map_G(k):
    return np.random.permutation(np.arange(1, k+1))


def f_i(G, std_dev):
    f_fn = np.vectorize(lambda g: (0.8 * g + 0.4 * math.ceil(g/2)))
    h = np.random.normal(0, std_dev, len(G))
    f = f_fn(G) + h

    return np.where(f < 0, 0, f)


class LearningAutomata:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def reinforce(self, beta):
        self.reward() if beta == 0 else self.penalize()

    def fit(self, std_dev=1, training_epochs=10000, testing_epochs=1000, bar=True, goal=0.85):

        G = map_G(self.n_actions)
        correct_count = 0
        training_count = 0
        wait = []

        converged_epochs = 0

        for i in tqdm(range(training_epochs + testing_epochs), disable=bar):
            floor_times = f_i(G, std_dev)
            optimal_floor = np.argmin(floor_times)
            pred = self.predict() - 1

            is_correct = floor_times[pred] == floor_times[optimal_floor]
            # cond = floor_times[pred] <= 1.3

            # beta=0 for reward, and beta=1 for penalty
            beta = int(not is_correct)
            training_count += int(is_correct)

            if i > 15 and training_count/i >= goal and converged_epochs==0:
                converged_epochs = i

            wait.append(floor_times[pred])
            self.reinforce(beta)

            if i >= training_epochs:
                correct_count += 1 if beta == 0 else 0

        accuracy = correct_count / testing_epochs
        ave_waiting_time = sum(wait) / len(wait)

        return [accuracy*100, ave_waiting_time, converged_epochs]

    def predict(self):
        pass

    def reward(self):
        pass

    def penalize(self):
        pass


class TsetlinAutomata(LearningAutomata):
    def __init__(self, n_actions, memory):
        super().__init__(n_actions)
        self.memory = memory
        random_action = np.random.randint(1, n_actions+1)
        self.state = random_action * memory - 1

    def predict(self):
        pred = self.state // self.memory + 1
        return pred

    def reward(self):
        if self.state % self.memory != 0:
            self.state -= 1

    def penalize(self):
        if self.state + 1 == self.memory * self.n_actions:
            self.state = self.memory - 1
        elif (self.state + 1) % self.memory == 0:
            self.state += self.memory
        else:
            self.state += 1


class KrinskyAutomata(LearningAutomata):
    def __init__(self, n_actions, memory):
        super().__init__(n_actions)
        self.memory = memory
        self.state = np.random.randint(1, n_actions+1) * memory - 1

    def predict(self):
        return self.state // self.memory + 1

    def reward(self):
        self.state -= self.state % self.memory

    def penalize(self):
        if self.state + 1 == self.memory * self.n_actions:
            self.state = self.memory - 1
        elif (self.state + 1) % self.memory == 0:
            self.state += self.memory
        else:
            self.state += 1


class KrylovAutomata(LearningAutomata):
    def __init__(self, n_actions, memory):
        super().__init__(n_actions)
        self.memory = memory
        self.state = np.random.randint(1, n_actions+1) * memory - 1

    def predict(self):
        return self.state // self.memory + 1

    def reward(self):
        if self.state % self.memory != 0:
            self.state -= 1

    def penalize(self):
        lucky = np.random.random() > 0.5
        if lucky:
            self.reward()
        else:
            if self.state + 1 == self.memory * self.n_actions:
                self.state = self.memory - 1
            elif (self.state + 1) % self.memory == 0:
                self.state += self.memory
            else:
                self.state += 1


class LRIAutomata(LearningAutomata):
    def __init__(self, n_actions, lamda):
        super().__init__(n_actions)
        self.p = np.full((n_actions,), 1/n_actions)
        self.lamda = lamda

    def predict(self):
        self.last_pred = np.random.choice(
            np.arange(1, self.n_actions+1), p=self.p)
        return self.last_pred

    def reward(self):
        last_pred = self.last_pred - 1
        self.p[last_pred] += self.lamda * (1 - self.p[last_pred])
        res = np.arange(self.n_actions) != last_pred
        self.p[res] -= self.lamda * self.p[res]

    def penalize(self):
        pass


def display_results(result):
    print(tabulate([result], headers=[
          'accuracy', 'waiting time (s)', 'convergence (epochs)']))


def run_model(model, name, std=1):
    print(f'\n\n=============== {name} ===============')
    results = model.fit(std_dev=std, bar=False,
                        training_epochs=10000, testing_epochs=1000)
    display_results(results)
    return results


def ensemble(model, memory, name, std=1, n_floors=6, n_experiments=100):
    print(f'\n\n=============== {name} ===============')
    results = []
    for i in tqdm(range(n_experiments)):
        auto_model = model(n_floors, memory)
        res = auto_model.fit(std, 5000, 1000)
        results.append(res)

    results = np.mean(np.array(results), axis=0)
    display_results(results)
    return results


def plot_bar(results):
    x = list(range(4))
    xlabels = ['Tsetlin', 'Krinsky', 'Krylov', 'LRI']
    ylabels = ['Accuracy (%)', 'Waiting time (s)', 'Number of epochs']
    titles = ['Accuracy of different automatas',
              'Average waiting time of different automatas',
              'Number of iterations to convergence']

    fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axarr[i].bar(x, results[:, i], color='blue')
        axarr[i].set_ylabel(ylabels[i])
        axarr[i].set_title(titles[i])
        axarr[i].set_xticks(x)
        axarr[i].set_xticklabels(xlabels, minor=False)

    plt.tight_layout()
    plt.show()


def run_models(std_dev):
    n_floors = 6
    results = []

    results.append(run_model(TsetlinAutomata(n_floors, 10),
                             'Tsetlin Automata', std=std_dev))
    results.append(run_model(KrinskyAutomata(n_floors, 10),
                             'Krinsky Automata', std=std_dev))
    results.append(run_model(KrylovAutomata(n_floors, 10),
                             'Krylov Automata', std=std_dev))
    results.append(run_model(LRIAutomata(n_floors, 0.2),
                             'LRI Automata', std=std_dev))

    plot_bar(np.array(results))


def run_ensemble(std_dev):
    results = []
    results.append(ensemble(TsetlinAutomata, 10,
                            'Tsetlin Automata', std=std_dev))
    results.append(ensemble(KrinskyAutomata, 10,
                            'Krinsky Automata', std=std_dev))
    results.append(ensemble(KrylovAutomata, 10,
                            'Krylov Automata', std=std_dev))
    results.append(ensemble(LRIAutomata, 0.2, 'LRI Automata', std=std_dev))
    plot_bar(np.array(results))
