from random import choice
import numpy as np
import random

# Environment Setup
class Player:
    
    def __init__(self, ball, x, y, pid):
        self.ball = ball
        self.x = x
        self.y = y
        self.pid = pid
    
#     def move(self, delta_x, delta_y):
#         self.x = self.x + delta_x
#         self.y = self.y + delta_y
        
class Soccer_Game:
    
    def __init__(self):
        self.nrow = 2
        self.ncol = 4
        self.playerA = Player(0, 0, 2, 0)  #initialize player A to be at position (0,2) without ball
        self.playerB = Player(1, 0, 1, 1)  #initialize player B to be at position (0,1) without ball
        self.goalACol = 0
        self.goalBCol = 3
        self.action_space = [0, 1, 2, 3, 4] # actions are N, S, E, W and Stick(Idle)
        self.done = 0
        # if choice([0,1]) == 0:
#             self.playerA.ball = 1
#         else:
#             self.playerB.ball = 1
            
    def reset(self):
        self.nrow = 2
        self.ncol = 4
        self.playerA = Player(0, 0, 2, 0)  #initialize player A to be at position (0,2) without ball
        self.playerB = Player(1, 0, 1, 1)  #initialize player B to be at position (0,1) without ball
        self.goalACol = 0
        self.goalBCol = 3
        self.action_space = [0, 1, 2, 3, 4] # actions are N, S, E, W and Stick(Idle)
        self.done = 0
        # if choice([0,1]) == 0:
#             self.playerA.ball = 1
#         else:
#             self.playerB.ball = 1
        
        
    def move(self, pid, direction):
        delta_x = 0
        delta_y = 0
        old_x = 0
        old_y = 0
        new_x = 0
        new_y = 0
        if direction == 0:
            delta_x = -1
        if direction == 1:
            delta_x = 1
        if direction == 2:
            delta_y = 1
        if direction == 3:
            delta_y = -1
        if pid == 0:
            old_x = self.playerA.x
            old_y = self.playerA.y
            new_x = self.playerA.x + delta_x
            new_y = self.playerA.y + delta_y
            # deal with situation when hitting boundary
            if (new_x > 1 or new_x < 0):
                new_x = self.playerA.x
                #print("wall hit")
            if (new_y > 3 or new_y < 0):
                new_y = self.playerA.y
                #print("wall hit")

            # deal with situation of collision
            if (new_x == self.playerB.x and new_y == self.playerB.y):
                new_x = self.playerA.x
                new_y = self.playerA.y
                #print("collide")

                # lose ball if currently holding it
                if self.playerA.ball == 1:
                    self.playerA.ball == 0
                    self.playerB.ball == 1
            self.playerA.x = new_x
            self.playerA.y = new_y
            if self.playerA.ball == 1:
                self.check_goal(0, new_y)
        else:
            old_x = self.playerB.x
            old_y = self.playerB.y
            new_x = self.playerB.x + delta_x
            new_y = self.playerB.y + delta_y
            # deal with situation when hitting boundary
            if (new_x > 1 or new_x < 0):
                new_x = self.playerB.x
            if (new_y > 3 or new_y < 0):
                new_y = self.playerB.y
            # deal with situation of collision
            if (new_x == self.playerA.x and new_y == self.playerA.y):
                new_x = self.playerB.x
                new_y = self.playerB.y
                # lose ball if currently holding it
                if self.playerB.ball == 1:
                    self.playerB.ball == 0
                    self.playerA.ball == 1
            self.playerB.x = new_x
            self.playerB.y = new_y
            if self.playerB.ball == 1:
                self.check_goal(1, new_y)
                
        r = self.get_reward()
            
        return (pid, old_x, old_y, new_x, new_y, r, self.done)
            
    def check_goal(self, pid, y):
        if pid == 0:
            if y == 0:
                #print("playerA scored A goal")
                self.done = 1
            elif y == 3:
                #print("playerA scored B goal")
                self.done = 2
        if pid == 1:
            if y == 0:
                #print("playerB scored A goal")
                self.done = 3
            elif y == 3:
                #print("playerB scored B goal")
                self.done = 4
                
    def get_reward(self):
        if self.done == 0:
            return [0,0]
        elif self.done == 1:
            return [100,-100]
        elif self.done == 2:
            return [-100, 100]
        elif self.done == 3:
            return [100, -100]
        else:
            return [-100, 100]
