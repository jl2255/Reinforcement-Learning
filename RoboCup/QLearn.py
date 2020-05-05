from random import choice
import numpy as np
import random
#from matplotlib import pyplot as plt
from Environment import Soccer_Game, Player
from plot import plot

def qTableInd():
    indTable = {}
    index = 0
    for i in range(2):
        for j in range(8):
            for m in range(8):
                if j!=m:
                    indTable[str(i) + str(j) + str(m)] = index
                    index = index + 1
    return indTable                

def decay(param):
    if param > 0.001:
        param = 0.999964 * param
    else:
        param = 0.001
    return param

# QLearning
def QLearn(game, gamma, alpha, epsilon, episodes):
           
    nrow = game.nrow
    ncol = game.ncol
    action_space = game.action_space
    naction = len(action_space)
    gamma = gamma
    alpha = alpha
    epsilon = epsilon
    q_list_A = []


        # total number of states are nrow*ncol*(nrow*ncol-1)*2. Every state consists of the A position, B position, 
        # who has the ball. In the q table, we locate state by: 1. row - 56 < 0 --> A, else --> B 2. (row - 56)/7 --> APos
        # if (row - 56)%7 < APos, (row - 56)%7 --> BPos, else, (row - 56)%7 + 1 --> BPos. 
        # ---------------------------------
        #  0    |    1    |    2    |    3
        # ---------------------------------
        #  4    |    5    |    6    |    7
        # ---------------------------------
        
    qInd = qTableInd()

    qTableA = np.zeros((112, naction))
    qTableB = np.zeros((112, naction))
    
    for e in range(episodes):
        game.reset()
        done = game.done

        # transform A state and B state to match q table
        cur_pos_A = game.playerA.x * 4 + game.playerA.y
        cur_pos_B = game.playerB.x * 4 + game.playerB.y
        #print("posA", cur_pos_A, "posB", cur_pos_B)
        
        ball = game.playerB.ball
        
        cur_state_A = qInd[str(ball) + str(cur_pos_A) + str(cur_pos_B)]
        #print cur_state_A
        cur_state_B = qInd[str(ball) + str(cur_pos_A) + str(cur_pos_B)]
        #print cur_state_A

        while done == 0:    
            # get current action
            if random.random() < epsilon:
                cur_a_A = choice(action_space)
                cur_a_B = choice(action_space)
            else:
                cur_a_A = np.argmax(qTableA[cur_state_A])
                cur_a_B = np.argmax(qTableB[cur_state_B])
                
            #print("actA", cur_a_A, "actB", cur_a_B)

            # move
            #print cur_a_B
            if random.random() < 0.5:
                pid_A, old_x_A, old_y_A, new_x_A, new_y_A, r_A, done_A = game.move(0, cur_a_A)
                pid_B, old_x_B, old_y_B, new_x_B, new_y_B, r_B, done_B = game.move(1, cur_a_B)
            else:
                pid_B, old_x_B, old_y_B, new_x_B, new_y_B, r_B, done_B = game.move(1, cur_a_B)
                pid_A, old_x_A, old_y_A, new_x_A, new_y_A, r_A, done_A = game.move(0, cur_a_A)

            # transform A state and B state to match q table
            #next_pos_A = new_x_A * 4 + new_y_A
            #next_pos_B = new_x_B * 4 + new_y_B
            
            next_pos_A = game.playerA.x * 4 + game.playerA.y
            next_pos_B = game.playerB.x * 4 + game.playerB.y
            
            #print("posA", next_pos_A, "posB", next_pos_B)
        
            ball = game.playerB.ball
        
            next_state_A = qInd[str(ball) + str(next_pos_A) + str(next_pos_B)]
            next_state_B = qInd[str(ball) + str(next_pos_A) + str(next_pos_B)]


            if done_A !=0 or done_B != 0:
                next_q_A = 0
                next_q_B = 0
            else:
                next_q_A = np.max(qTableA[next_state_A])
                next_q_B = np.max(qTableB[next_state_B])
                            
            #update q table    
            qTableA[cur_state_A,cur_a_A] = qTableA[cur_state_A,cur_a_A] + alpha * ((1-gamma)*(r_A[0] + r_B[0]) + gamma * next_q_A - qTableA[cur_state_A,cur_a_A])
            qTableB[cur_state_B,cur_a_B] = qTableB[cur_state_B,cur_a_B] + alpha * ((1-gamma)*(r_A[1] + r_B[1]) + gamma * next_q_B - qTableB[cur_state_B,cur_a_B])
            
            cur_state_A = next_state_A
            cur_state_B = next_state_B

            done = done_A + done_B
            
            q_list_A.append(qTableA[71,1]) 
        
        epsilon = decay(epsilon)
        alpha = decay(alpha)

        #if e%10000==0:
        print(str(e) + "completed, " + "alpha: " + str(alpha) )
            
    return q_list_A, qTableA, qTableB
    
game = Soccer_Game()
q_list_A, qTableA, qTableB = QLearn(game, 0.9, 1.0, 1.0, 150000)
np.save("QLearn_q_a_list", q_list_A)
np.save("QLearn_qTableA", qTableA)
np.save("QLearn_qTableB", qTableB)
plot(np.array(q_list_A), "Q-learner", "Q-learner")


