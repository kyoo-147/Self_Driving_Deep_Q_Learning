# -*- coding: utf-8 -*-
# Contatct: AI-Lab - Smart Things
# AI for Self Driving Car

# Nhập các thư viện cần
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Khởi tạo một kiến trúc của mạng thần kinh Neural
class Network(nn.Module):    
    def __init__(self, input_size, nb_action):
        # Khởi tạo kiến trúc nơ-ron tương ứng - kiến trúc có một lớp đầu vào với số lương nơ-ron tương ứng với input_size
        super(Network, self).__init__()
        self.input_size = input_size
        # khỏi tạo nb_action dùng để ước tính giá trị Q cho hành động
        self.nb_action = nb_action
        # ở đây ta có thể tăng kích thước mạng nhằm mô hình học được phức tạp hơn
        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, nb_action)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Thực hiện phát lại kinh nghiệmm - lặp lại bộ nhớ
# Nhằm tái sử dụng dữ liệu trước - để cải thiện hiệu suất học tập
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Thực hiện triển khai thuật toán Deep Q-Learning
class Dqn():    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        # Lặp lại bộ nhớ với 100 ngàn lần lặp
        self.memory = ReplayMemory(100000)
        # Khởi tạo trình tối ưu hóa sử dụng bằng thuật toán ADAM 
        # Kết hợp hai kỹ thuật quan trọng - Gradient Descent + Momentum với learning rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    # Lựa chọn hành động dựa trên mạng neural hiện tại và chính sách epsilon-greedy
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial(1)
        return action.data[0,0]
    
    # Thực hiện quá trình học của mạng neural dựa trên dữ liệu thu thập và tính toán hoàn mất mát
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Sử dụng mạng neural model dự đoán giá trị q hiện tại sau đó thực hiện batch_action tương ứng
        # Để lựa chọn giá trị Q tương ứng
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Được sử dụng để lấy giá trị Q cao nhất
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        # Tính toán hàm mất mát sử dụng L1 giữa giá trị Q ước tính và giá trị mục tiêu
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    # Cập nhập trạng thái hiện tại của hệ thống sau mỗi bước và thực hiện việc học
    def update(self, reward, new_signal):
        # Chuyển dổi new_signal thành đối tượng Tensor
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # Cập nhập trạng thái mới và cũ, hành động đã thực hiện và phần thưởng vào bộ nhớ kinh nghiệm 
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        # Nếu bộ nhớ đã đủ lớn hơn 100, thực hiện quá trình học
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        # khởi tạo hành động cuối cùng thành hành động mới
        self.last_action = action
        # khởi tạo trạng thái cuối thành trạng thái mới
        self.last_state = new_state
        self.last_reward = reward
        # Thêm phần thưởng mới vào cửa sổ của các phần thưởng
        self.reward_window.append(reward)
        # Nếu phần thưởng lớn hơn 1000, chương trình loại bỏ phần tử đầu tiên để duy trì kích thước
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    # Tính điểm số hiện tại của hệ thống dựa trên thưởng nhận được
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    # Lưu trọng số neural
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    # Tải ccacs trọng số neural
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")