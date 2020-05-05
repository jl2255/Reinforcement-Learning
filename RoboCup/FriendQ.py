import numpy as np
from Environment import Soccer_Game, Player
from random import choice
import random
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
        param = 0.9998 * param
    else:
        param = 0.001
    return param

def FriendQ(game, gamma, alpha, episodes):
    qTable = np.zeros((112, 5, 5))
    q_error = []
    action_space = game.action_space
    naction = len(action_space)
    qInd = qTableInd()

    for e in range(episodes):
        game.reset()
        done = 0

        # transform A state and B state to match q table
        cur_pos_A = game.playerA.x * 4 + game.playerA.y
        cur_pos_B = game.playerB.x * 4 + game.playerB.y
        
        ball = game.playerB.ball
        
        cur_state_A = qInd[str(ball) + str(cur_pos_A) + str(cur_pos_B)]
        cur_state_B = qInd[str(ball) + str(cur_pos_A) + str(cur_pos_B)]
        
        while done == 0:
            cur_a_A = choice(action_space)
            cur_a_B = choice(action_space)
            
            if random.random() < 0.5:
                pid_A, old_x_A, old_y_A, new_x_A, new_y_A, r_A, done_A = game.move(0, cur_a_A)
                pid_B, old_x_B, old_y_B, new_x_B, new_y_B, r_B, done_B = game.move(1, cur_a_B)
            else:
                pid_B, old_x_B, old_y_B, new_x_B, new_y_B, r_B, done_B = game.move(1, cur_a_B)
                pid_A, old_x_A, old_y_A, new_x_A, new_y_A, r_A, done_A = game.move(0, cur_a_A)
                
            # transform A state and B state to match q table
            next_pos_A = game.playerA.x * 4 + game.playerA.y
            next_pos_B = game.playerB.x * 4 + game.playerB.y
        
            ball = game.playerB.ball
        
            next_state_A = qInd[str(ball) + str(next_pos_A) + str(next_pos_B)]
            next_state_B = qInd[str(ball) + str(next_pos_A) + str(next_pos_B)]

            if done_A != 0 or done_B !=0:
                max_action = 0
            else:
                max_action = np.max(qTable[next_state_A])

            #print len(qTable)

            qTable[cur_state_A,cur_a_A,cur_a_B] = qTable[cur_state_A,cur_a_A,cur_a_B] + alpha * ((r_A[0] + r_B[0]) + gamma * max_action - qTable[cur_state_A,cur_a_A,cur_a_B])

            cur_state_A = next_state_A
            cur_state_B = next_state_B

            done = done_A + done_B
            
            q_error.append(qTable[71, 1, 4])

        alpha = decay(alpha)
        
        if e%10000==0:
        	print("Episode number ", e, " Alpha ", alpha)        

    return q_error, qTable

game = Soccer_Game()
q_a_list, qTable  = FriendQ(game, 0.9, 0.03, 1500000)
np.save("FriendQ_q_a_list", q_a_list)
np.save("FriendQ_qTable", qTable)
plot(np.array(q_a_list), "Friend-Q", "Friend-Q")

