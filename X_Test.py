# -*- coding: utf-8 -*-
# Contatct: AI-Lab - Smart Things
# Self Driving Car

# Nhập các thư viện cần thiết 
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Nhập module package của Kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Nhập thuật toán Deep Q-Learning
from X_Ai import Dqn

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# khởi tạo 2 biến để lưu trữ điểm cuối cùng trong bộ nhớ khi vẽ lên bản đồ
last_x = 0
last_y = 0
n_points = 0 # tổng số điểm tại lần cuối vẽ
length = 0 # chiều dài tại lần cuối vẽ

# Gọi Dqn và truyền tham số vào
brain = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama = 0.9
action2rotation = [0,20,-20] # hành động = 0 => không quay, hành động = 1 => xoay 20 độ, hành động = 2 => xoay -20 độ
last_reward = 0 # Khởi tạo phần thưởng cuối cùng
scores = [] # khởi tạo đường cong điểm trung bình (cửa sổ trượt phần thưởng) theo thời gian

# Khởi tạo map
first_update = True 
def init():
    global sand 
    global goal_x # x-tọa độ của mục tiêu (nơi ô tô phải đi, đó là sân bay hoặc trung tâm thành phố)
    global goal_y # tọa độ y của mục tiêu (nơi ô tô phải đi, đó là sân bay hoặc trung tâm thành phố)
    # khởi tạo mảng bằng 0 
    sand = np.zeros((longueur,largeur)) 
    # mục tiêu cần đạt là ở phía trên bên trái bản đồ (tọa độ x là 20 chứ không phải 0 vì xe sẽ nhận được phần thưởng xấu nếu chạm vào tường)
    goal_x = 20 
    # mục tiêu cần đạt được nằm ở phía trên bên trái bản đồ (tọa độ y)
    goal_y = largeur - 20
    # Khởi tạo map một lần 
    first_update = False 

# Khởi tạo khoảng cách cuối cùng
last_distance = 0

class Car(Widget):
    # khởi tạo góc của ô tô (góc giữa trục x của bản đồ và trục của ô tô)
    angle = NumericProperty(0) 
    # khởi tạo góc của ô tô (góc giữa trục x của bản đồ và trục của ô tô)
    rotation = NumericProperty(0)
    # khởi tạo tọa độ x của vectơ vận tốc
    velocity_x = NumericProperty(0) 
    # khởi tạo tọa độ y của vectơ vận tốc
    velocity_y = NumericProperty(0) 
    # vector vận tốc
    velocity = ReferenceListProperty(velocity_x, velocity_y) 
    # khởi tạo tọa độ x của cảm biến đầu tiên (cảm biến nhìn về phía trước)
    sensor1_x = NumericProperty(0) 
    # khởi tạo tọa độ y của cảm biến đầu tiên (cảm biến nhìn về phía trước)
    sensor1_y = NumericProperty(0) 
    # vector cảm biến đầu tiên
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # first sensor vector
    # khởi tạo tọa độ x của cảm biến thứ hai (cái nhìn sang trái 30 độ)
    sensor2_x = NumericProperty(0) 
    # khởi tạo tọa độ y của cảm biến thứ hai (cái nhìn sang trái 30 độ)
    sensor2_y = NumericProperty(0) 
    # vector cảm biến thứ hai
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) 
     # khởi tạo tọa độ x của cảm biến thứ ba (cái nhìn sang phải 30 độ)
    sensor3_x = NumericProperty(0) 
     # khởi tạo tọa độ y của cảm biến thứ ba (cái nhìn sang phải 30 độ)
    sensor3_y = NumericProperty(0)
    # vector cảm biến thứ ba
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) 
    # khởi tạo tín hiệu mà cảm biến 1 nhận được
    signal1 = NumericProperty(0) 
    # khởi tạo tín hiệu mà cảm biến 2 nhận được
    signal2 = NumericProperty(0) 
    # khởi tạo tín hiệu mà cảm biến 3 nhận được
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # cập nhật vị trí của xe theo vị trí và vận tốc cuối cùng của nó
        self.pos = Vector(*self.velocity) + self.pos 
        # bắt đầu vòng quay của xe
        self.rotation = rotation 
        # cập nhập góc
        self.angle = self.angle + self.rotation 
        # cập nhật vị trí cảm biến 1
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos 
        # cập nhật vị trí cảm biến 2
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos 
        # cập nhật vị trí cảm biến 3
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos 
        # nhận tín hiệu mà cảm biến 1 nhận được (mật độ cát xung quanh cảm biến 1)
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        # nhận tín hiệu mà cảm biến 2 nhận được (mật độ cát xung quanh cảm biến 2)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400. 
        # nhận tín hiệu mà cảm biến 3 nhận được (mật độ cát xung quanh cảm biến 3)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400. 
        # nếu cảm biến 1 nằm ngoài bản đồ (xe đang hướng về một cạnh của bản đồ)
        if self.sensor1_x > longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            
            self.signal1 = 1.
        # nếu cảm biến 2 nằm ngoài bản đồ (xe đang hướng về một cạnh của bản đồ)
        if self.sensor2_x > longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            # cảm biến 2 phát hiện đầy đủ sand 
            self.signal2 = 1. 
        # nếu cảm biến 3 nằm ngoài bản đồ (xe đang hướng về một cạnh của bản đồ)
        if self.sensor3_x > longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10: 
            # cảm biến 3 phát hiện đầy đủ sand 
            self.signal3 = 1. 

class Ball1(Widget): # sensor 1 
    pass
class Ball2(Widget): # sensor 2 
    pass
class Ball3(Widget): # sensor 3 
    pass

class Game(Widget):
    # bắt tạo vật thể xe từ tệp kivy
    car = ObjectProperty(None) 
    # bắt đầu cảm biến vật thể 1
    ball1 = ObjectProperty(None) 
    # bắt đầu cảm biến vật thể 2
    ball2 = ObjectProperty(None)
    # bắt đầu cảm biến vật thể 3
    ball3 = ObjectProperty(None) # getting the sensor 3 object from our kivy file

    def serve_car(self): 
        # xe sẽ được khởi tạo tại giữa map
        self.car.center = self.center 
        # xe sẽ bắt đầu đi ngang về bên phải với tốc độ 6
        self.car.velocity = Vector(6, 0)

    def update(self, dt): 
    # chức năng cập nhật lớn cập nhật mọi thứ cần cập nhật vào từng thời điểm t riêng biệt khi đạt đến trạng thái mới (nhận tín hiệu mới từ các cảm biến)
        global brain  # chỉ định các biến toàn cục (bộ não của ô tô, đó là AI của chúng ta)
        global last_reward # chỉ định các biến toàn cục (phần thưởng cuối cùng)
        global scores # # chỉ định các biến toàn cục (phương tiện của phần thưởng)
        global last_distance # chỉ định các biến toàn cục (khoảng cách cuối cùng từ ô tô đến mục tiêu)
        global goal_x # chỉ định các biến toàn cục (tọa độ x của mục tiêu)
        global goal_y # chỉ định các biến toàn cục (tọa độ y của mục tiêu)
        global longueur # chỉ định các biến toàn cục (chiều rộng của bản đồ)
        global largeur # chỉ định các biến toàn cục (chiều cao của bản đồ)

        # chiều rộng của bản đồ (cạnh ngang)
        longueur = self.width 
        # chiều cao của bản đồ (cạnh dọc)
        largeur = self.height 
        if first_update:
            init()

        # chênh lệch tọa độ x giữa mục tiêu và ô tô
        xx = goal_x - self.car.x 
        # chênh lệch tọa độ y giữa mục tiêu và ô tô
        yy = goal_y - self.car.y 
        # hướng của ô tô so với mục tiêu (nếu ô tô đang hướng hoàn toàn về phía mục tiêu thì hướng = 0)
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        # vectơ trạng thái đầu vào, bao gồm ba tín hiệu mà ba cảm biến nhận được, cộng với hướng và hướng
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] 
        # thực hiện hành động từ AI (bộ não đối tượng của lớp dqn)
        action = brain.update(last_reward, last_signal) 
        # thêm điểm (trung bình của 100 phần thưởng cuối cùng vào cửa sổ phần thưởng)
        scores.append(brain.score()) 
        # chuyển đổi hành động được thực hiện (0, 1 hoặc 2) thành góc quay (0°, 20° hoặc -20°)
        rotation = action2rotation[action] 
        # di chuyển xe theo góc quay cuối cùng này
        self.car.move(rotation) 
        # đạt được khoảng cách mới giữa xe và đích ngay sau khi xe di chuyển
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) 
        # cập nhật vị trí cảm biến đầu tiên (ball1) ngay sau khi xe di chuyển
        self.ball1.pos = self.car.sensor1 
        # cập nhật vị trí của cảm biến thứ hai (ball2) ngay sau khi xe di chuyển
        self.ball2.pos = self.car.sensor2
        # cập nhật vị trí của cảm biến thứ hai (ball3) ngay sau khi xe di chuyển
        self.ball3.pos = self.car.sensor3 

        # nếu xe chạy trên cát
        if sand[int(self.car.x),int(self.car.y)] > 0: 
            # nó bị chậm lại (tốc độ = 1)
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) 
            # và giảm phần thưởng -1
            last_reward = -1 
        else: 
            # nó chuyển sang tốc độ bình thường (tốc độ = 6)
            self.car.velocity = Vector(6, 0).rotate(self.car.angle) 
            # và nó nhận được phần thưởng xấu (-0,2)
            last_reward = -0.2 
            # tuy nhiên nếu nó tiến gần đến mục tiêu
            if distance < last_distance: 
                # vẫn nhận được phần thưởng tích cực 0,1
                last_reward = 0.1

        # nếu xe ở mép trái khung
        if self.car.x < 10: 
            # nó không bị chậm lại
            self.car.x = 10 
            last_reward = -1 
        if self.car.x > self.width-10: 
            # nếu xe ở mép phải của khung
            self.car.x = self.width-10 
            last_reward = -1 
        # nếu xe ở mép dưới của khung
        if self.car.y < 10: 
            self.car.y = 10 
            last_reward = -1 
        # nếu xe ở mép trên của khung
        if self.car.y > self.height-10: 
            self.car.y = self.height-10 
            last_reward = -1

        # khi xe đạt mục tiêu
        if distance < 100:
            # mục tiêu trở thành góc dưới bên phải của bản đồ (trung tâm thành phố) và ngược lại (cập nhật tọa độ x của mục tiêu)
            goal_x = self.width - goal_x 
            # mục tiêu trở thành góc dưới bên phải của bản đồ (trung tâm thành phố) và ngược lại (cập nhật tọa độ y của mục tiêu)
            goal_y = self.height - goal_y 
        # Cập nhật khoảng cách cuối cùng từ xe đến mục tiêu
        last_distance = distance

class MyPaintWidget(Widget):
    # Tùy chỉnh nhé vì tôi lười ghi quá rồi (❁´◡`❁)
    def on_touch_down(self, touch): 
        global length,n_points,last_x,last_y
        with self.canvas:
            Color(0.5,0.8,0)
            d=10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    # thực hiện hiển thị khi bắt đầu thao tác vẽ đường đi lên giao diện
    def on_touch_move(self, touch): 
        global length,n_points,last_x,last_y
        if touch.button=='left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

class CarApp(App):
    # Xây dựng app
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='Xóa dữ liệu')
        savebtn = Button(text='Lưu model',pos=(parent.width,0))
        loadbtn = Button(text='Tải Model',pos=(2*parent.width,0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj): 
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj): 
        print("loading last saved brain...")
        brain.load()

if __name__ == '__main__':
    CarApp().run()
