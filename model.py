import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        print('*** Linear_QNet init ***')
        self.linear1 = nn.Linear(input_size, hidden_size) # First/Hidden layer
        self.linear2 = nn.Linear(hidden_size, output_size) # Output layer


    def forward(self, x):
        x = F.relu(self.linear1(x)) # First layer
        return self.linear2(x) # Output layer

    def save(self, file_name='model.pth'):
        print('>>>> Saving env ...', end=" ")
        model_folder_path = os.path.join(os.getcwd(), 'model')
        file_path = os.path.join(model_folder_path, file_name)

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            torch.save(self.state_dict(), file_path)
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
                torch.save(self.state_dict(), file_path)
                print(f'The weights file has been deleted.New weights file created!!.')
            else:
                torch.save(self.state_dict(), file_path)
                print(f'The weights file did not exist. New weights file created!!')



    def load_model(self, file_name='model.pth'):
        print('+++ Loading agent model...', end=" ")
        model_folder_path = os.path.join(os.getcwd(), 'model')
        weights_file = os.path.join(model_folder_path, file_name)

        if os.path.exists(weights_file):
            self.load_state_dict(torch.load(weights_file))
            print("Weights loaded successfully!!.")
        else:
            print(f"WARNING: The weights file does not exist. No weights loaded.")

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.data} | {param.size()}")

class QTrainer:
    def __init__(self, model, lr, gamma):
        print('*** QTrainer init ***')
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # We need to convert all of them to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # We check if the state is a single state or a batch of states
        if len(state.shape) == 1:
            # Size (1, x)
            # unsqueeze turns an n.d. tensor into an (n+1).d. one by adding an extra dimension of depth 1.
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)


        # 1: predicted Q values with current state
        predicted_q = self.model(state)
        # pred.clone()
        target = predicted_q.clone()

        for idx in range(len(done)):
            if not done[idx]:
                # 2: Q_new = r + gamma * max(next_predicted Q value) -> only do this if not done
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            else:
                Q_new = reward[idx]

            # preds[argmax(action)] = Q_new
            target[idx][torch.argmax(action[idx]).item()] = Q_new


        self.optimizer.zero_grad() # Reset the gradients from the previous iteration
        loss = self.criterion(target, predicted_q) # Calculate the loss
        loss.backward() # Calculate the gradients

        self.optimizer.step()


