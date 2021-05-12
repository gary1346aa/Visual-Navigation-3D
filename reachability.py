import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import random

import cv2
import glm
import pyrender
import scene_render
import maze

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 512
ALPHA = 1
BETA = 0.5
NOVELTY_THRESHOLD = 0.

class R_Network(nn.Module):

    def __init__(self, pretrained=False):

        super().__init__()

        self.embedding_network = nn.Sequential(
            *list(models.resnet18(pretrained=pretrained).children())[:-1])

        self.similarity_network = nn.Sequential(
            nn.BatchNorm1d(2*EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(2*EMBEDDING_DIM, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

    def embed_observation(self, x):
        x_emb = self.embedding_network(x)
        x_emb = x_emb.reshape(x_emb.shape[0], -1)
        return x_emb

    def compute_similarity(self, x1_emb, x2_emb):
        input = torch.cat((x1_emb, x2_emb), dim=-1)
        return self.similarity_network(input)

    def compute_loss(self, x1, x2, y):
        x1_emb = self.embed_observation(x1)
        x2_emb = self.embed_observation(x2)
        y_pred = self.compute_similarity(x1_emb, x2_emb)
        loss = self.loss_fn(y_pred, y)
        return loss

    def get_params(self):
        return list(self.embedding_network.parameters()) + \
               list(self.similarity_network.parameters())

    def save_model(self):
        torch.save(self.similarity_network, "sim_{self.exp_name}")
        torch.save(self.embedding_network, "emb_{self.exp_name}")

    def load_model(self, eval=False):
        self.similarity_network = torch.load(f'sim_{self.exp_name}')
        self.embedding_network = torch.load(f'emb_{self.exp_name}')
        if eval:
            self.similarity_network.eval()
            self.embedding_network.eval()

class R_Network_Trainer(object):

    def __init__(self,
                 r_network,
                 buffer_size=20000,
                 training_interval=200000,
                 num_epochs=6,
                 exp_name=None):

        self._r_network = r_network
        self._batch_size = 64
        self._optimizer = Adam(self._r_network.get_params(), lr=1e-4)
        self._training_interval = training_interval
        self._num_epochs = num_epochs

        self._buffer_states = [None] * buffer_size
        self._buffer_dones = [None] * buffer_size
        self._buffer_index = 0
        self._buffer_count = 0

        self._exp_name = exp_name
        self._log_train_loss = []
        self._log_valid_acc = []

    def store_new_state(self, state, reward, done, info=None):

        self._buffer_states[self._buffer_index] = state.copy()
        self._buffer_dones[self._buffer_index] = done
        self._buffer_index = (self._buffer_index + 1) % len(self._buffer_states)
        self._buffer_count += 1

        if self._buffer_count == 20000 or \
           (self._buffer_count > 0 and \
           self._buffer_count % self._training_interval == 0):
            history_states, history_dones = self._get_flatten_history()
            self.train(history_states, history_dones)
    
    def _get_flatten_history(self):
        
        if self._buffer_count < len(self._buffer_states):
            return self._buffer_states[:self._buffer_count], \
                    self._buffer_dones[:self._buffer_count]

        # Reorder the indices.
        history_states = self._buffer_states[self._buffer_index:]
        history_states.extend(self._buffer_states[:self._buffer_index])
        history_dones = self._buffer_dones[self._buffer_index:]
        history_dones.extend(self._buffer_dones[:self._buffer_index])
        return history_states, history_dones

    def _split_history(self, states, dones):

        if len(states) == 0:
            return []

        nenvs = len(dones[0])
        nsteps = len(dones)

        start_index = [0] * nenvs

        trajectories = []
        for k in range(nsteps):
            for n in range(nenvs):
                if dones[k][n] == True or k == nsteps - 1:
                    next_start_index = k + 1
                    time_slice = states[start_index[n]:next_start_index]
                    trajectories.append([state[n] for state in time_slice])
                    start_index[n] = next_start_index

        return trajectories

    def _prepare_data(self, states, dones):

        max_action_distance = 5
        
        all_x1 = []
        all_x2 = []
        all_y = []
        trajectories = self._split_history(states, dones)

        for trajectory in trajectories:
            x1, x2, y = self._create_training_data(trajectory, max_action_distance)
            all_x1.extend(x1)
            all_x2.extend(x2)
            all_y.extend(y)

        return all_x1, all_x2, all_y

    def _create_training_data(self, trajectory, max_action_distance):

        first_second_y = []
        buffer_position = 0

        while True:
            positive_example_canditate = \
                buffer_position + random.randint(1, max_action_distance)
            next_buffer_position = buffer_position + random.randint(1, 5) + 1

            if next_buffer_position >= len(trajectory) or \
               positive_example_canditate >= len(trajectory):
               break

            y = random.randint(0, 1)

            if y == 1: # Create positive example
                if random.random() < 0.5:
                    first, second = buffer_position, next_buffer_position
                else:
                    second, first = buffer_position, next_buffer_position
            elif y == 0:  # Create negative example
                assert buffer_position < len(trajectory)
                time_interval = 5 * max_action_distance
                min_index = max(buffer_position - time_interval, 0)
                max_index = min(buffer_position + time_interval + 1, len(trajectory))

                effective_length = len(trajectory) - (max_index - min_index)
                range_max = effective_length - 1
                if range_max <= 0:
                    return buffer_position, None
                index = random.randint(0, range_max)
                if index >= min_index:
                    index += (max_index - min_index)
                first, second = buffer_position, index

            if first is None or second is None:
                break
            
            first_second_y.append((first, second, y))
            buffer_position = next_buffer_position

        x1 = []
        x2 = []
        ys = []
        for first, second, y in first_second_y:
            x1.append(trajectory[first])
            x2.append(trajectory[second])
            ys.append(y)
        return x1, x2, ys

    def _shuffle(self, x1, x2, y):
        
        sample_count = len(x1)
        assert len(x2) == sample_count
        assert len(y) == sample_count
        
        permutation = np.random.permutation(sample_count)
        x1 = [x1[p] for p in permutation]
        x2 = [x2[p] for p in permutation]
        y = [y[p] for p in permutation]
        return x1, x2, y

    def _generate_batch(self, x1, x2, y):

        while True:
            sample_count = len(x1)
            num_batches = sample_count // self._batch_size
            for batch_index in range(num_batches):
                from_index = batch_index * self._batch_size
                to_index = from_index + self._batch_size
                yield [np.array(x1[from_index:to_index]), 
                       np.array(x2[from_index:to_index]),
                       np.array(y[from_index:to_index])]

            x1, x2, y = self._shuffle(x1, x2, y)

    def train(self, history_states, history_dones):

        x1, x2, y = self._prepare_data(history_states, history_dones)
        x1, x2, y = self._shuffle(x1, x2, y)

        n = len(x1)
        train_count = (n * 95) // 100
        print('Train_count = ', train_count)

        x1_train, x2_train, y_train = \
            x1[:train_count], x2[:train_count], y[:train_count]
        x1_valid, x2_valid, y_valid = \
            x1[train_count:], x2[train_count:], y[train_count:]

        validation_data = [np.array(x1_valid), 
                           np.array(x2_valid),
                           np.array(y_valid)]

        batch_generator = self._generate_batch(x1_train, x2_train, y_train)

        for epoch in range(self._num_epochs):
            for batch in range(train_count // self._batch_size):

                x1_batch, x2_batch, y_batch = next(batch_generator)
                x1_batch = torch.tensor(x1_batch).permute(0, 3, 1, 2).to(device)
                x2_batch = torch.tensor(x2_batch).permute(0, 3, 1, 2).to(device)
                y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).to(device)

                self._optimizer.zero_grad()
                loss = self._r_network.compute_loss(x1_batch, x2_batch, y_batch)
                loss.backward()
                self._optimizer.step()

                print(f'Epoch #{epoch+1}/{self._num_epochs} - Batch {batch:2d}/{train_count // self._batch_size}: loss = {loss:.3f}')
                self._log_train_loss.append(loss)

            acc, cnt = 0, 0
            for batch in range(len(validation_data[0]) // self._batch_size):

                x1_valid = torch.tensor(validation_data[0][batch*self._batch_size:(batch+1)*self._batch_size]).permute(0, 3, 1, 2).to(device)
                x2_valid = torch.tensor(validation_data[1][batch*self._batch_size:(batch+1)*self._batch_size]).permute(0, 3, 1, 2).to(device)
                y_valid = torch.FloatTensor(validation_data[2][batch*self._batch_size:(batch+1)*self._batch_size]).unsqueeze(-1).to(device)

                x1_valid_emb = self._r_network.embed_observation(x1_valid)
                x2_valid_emb = self._r_network.embed_observation(x2_valid)
                y_pred = self._r_network.compute_similarity(x1_valid_emb, x2_valid_emb)

                acc += (abs(y_valid-y_pred) < 0.5).sum().item()
                cnt += self._batch_size

            if cnt > 0:
                print(f'Epoch #{epoch+1}/{self._num_epochs} - Validation: acc = {acc/cnt:.4f}')
                self._log_valid_acc.append(acc/cnt)

    def loss_and_validation(self):
        
        t = np.arange(0.0, 60.0, 60.0/len(self._log_train_loss))
        
        fig, ax = plt.subplots()
        ax.plot(t, self._log_train_loss)

        ax.set(xlabel="epochs", ylabel="loss", title="training loss")
        ax.grid()

        fig.savefig(f"Training_{self._exp_name}.png")
        plt.show()

        t = np.arange(0.0, 60.0, 60.0/len(self._log_valid_acc))
        
        fig, ax = plt.subplots()
        ax.plot(t, self._log_valid_acc)

        ax.set(xlabel="epochs", ylabel="accuracy", title="validation accuracy")
        ax.grid()

        fig.savefig(f"Validation_{self._exp_name}.png")
        plt.show()


class EpisodicMemory(object):

    def __init__(self, 
                 embedding_shape,    # the shape should be list
                 capacity=200, 
                 replacement='random', 
                 ):

        if replacement not in ['fifo', 'random']:
            raise ValueError("Invalid replacement method.")        

        self._capacity = capacity
        self._replacement = replacement
        self._embedding_shape = embedding_shape

        self.reset()
        
    def reset(self):
        
        self._count = 0
        # numpy array can store torch tensor
        self._memory = np.zeros([self._capacity] + self._embedding_shape)
        self._memory_age = np.zeros([self._capacity], dtype=np.int32)

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return min(self._count, self._capacity)

    def store_new_state(self, state_embedding):

        if self._count >= self._capacity:
            if self._replacement == 'fifo':
                index = self._count % self._capacity
            elif self._replacement == 'random':
                index = np.random.randint(low=0, high=self._capacity)
        else:
            index = self._count

        self._memory[index] = state_embedding.copy()
        self._memory_age[index] = self._count
        self._count += 1

        return index

    def get_similarity_buffer(self, state_embedding, r_network):

        state_embedding = np.array([state_embedding] * len(self))
        state_embedding = torch.FloatTensor(state_embedding).to(device)
        memory = torch.FloatTensor(self._memory[:len(self)]).to(device)

        similarity_buffer = r_network.compute_similarity(state_embedding, memory).cpu().numpy()
        
        return similarity_buffer


def similarity_to_memory(state_embedding, episodic_memory, r_network, percentile=90):

    if len(episodic_memory) == 0:
        return 0.0, 0.0, 0.0

    similarity_buffer = episodic_memory.get_similarity_buffer(state_embedding, r_network)
    aggregated = np.percentile(similarity_buffer, percentile)
    
    return aggregated, np.amin(similarity_buffer), np.amax(similarity_buffer)


if __name__ == "__main__":

    from maze_env import MazeBaseEnv
    import maze
    maze_obj = maze.MazeBoardRandom()
    env = MazeBaseEnv(maze_obj, render_res=(192, 192))
    env.reset()

    action_space = [0, 1, 2, 3]
    prob_l = [0.6, 0.2, 0.1, 0.1]
    prob_r = [0.5, 0, 0.1, 0]

    r_network = R_Network().to(device)
    trainer = R_Network_Trainer(r_network=r_network, exp_name="pg")
    episodic_memory = EpisodicMemory(embedding_shape=[EMBEDDING_DIM])
    state, info = env.reset()

    step = 0
    while(True):

        if step % 200 > 100 and step % 200 < 200:
            action = random.choices(action_space, prob_l)[0]
        else:
            action = random.choices(action_space, prob_r)[0]

        state_next, reward, done, info = env.step(action)
        state_next = state_next.astype(np.float32) / 255.
        step += 1
        if step > 0 and step % 500 == 0:
            done = True
        trainer.store_new_state([state], [reward], [done], [info])

        if step > 40000:
            r_network.eval()
            with torch.no_grad():
                state_embedding = r_network.embed_observation(torch.FloatTensor([state]).permute(0, 3, 1, 2).to(device)).cpu().numpy()[0]
                # episodic_memory.store_new_state(state_embedding)
                aggregated, min_value, max_value = similarity_to_memory(state_embedding, episodic_memory, r_network)
                curiosity_bonus = ALPHA * (BETA - aggregated)
                if curiosity_bonus > NOVELTY_THRESHOLD or len(episodic_memory) == 0:
                    episodic_memory.store_new_state(state_embedding)
            r_network.train()
            print(f"{aggregated:.3f}, {min_value:.3f}, {max_value:.3f}, {curiosity_bonus:.3f}, {len(episodic_memory)}")

        state = state_next.copy()
        if step > 0 and step % 1000 == 0:
            print(f'step count = {step}')

        if step > 0 and step % 500 == 0:
            state, info = env.reset(gen_maze=True)

        if step == 200000:
            trainer._r_network.save_model()
            trainer.loss_and_validation()
            break

        