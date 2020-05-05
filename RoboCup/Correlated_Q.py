import numpy as np
from Environment import Soccer_Game, Player
from random import choice
import random
#from matplotlib import pyplot as plt
from cvxopt import matrix,solvers
from plot import plot
#%matplotlib inline

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
        param = 0.99995 * param
    else:
        param = 0.001
    return param                          

def CEQ(game, gamma, alpha, episodes):
    
    qTableA = np.ones((112, 5, 5))
    qTableB = np.ones((112, 5, 5))
    q_error = []
    q_error1 = []
    q_error2 = []
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
            
            # constraints
            qA = qTableA[next_state_A]
            constrAll =[]
            for i in range(5):
                constr = []
                for j in range(5):
                    if i != j:
                        constr.append(qA[i,:] - qA[j,:])
                constrAll.append(constr)

            constrA = np.zeros((20,25))
            constrA[:4,:5] = constrAll[0]
            constrA[4:8,5:10] = constrAll[1]
            constrA[8:12,10:15] = constrAll[2]
            constrA[12:16, 15:20] = constrAll[3]
            constrA[16:20,20:25] = constrAll[4]


            qB = qTableB[next_state_B]
            constrAll = []
            for i in range(5):
                constr = []
                for j in range(5):
                    if i != j:
                        constr.append(qB[:, i] - qB[:, j])
                constrAll.append(constr)

            constrB = np.zeros((20, 25))
            
            for j in [0, 5, 10, 15, 20]:
                for i in [0, 1, 2, 3]:
                    constrB[i, j] = constrAll[0][i][j/5]
            
            for j in [1, 6, 11, 16, 21]:
                for i in [4, 5, 6, 7]:
                    constrB[i, j] = constrAll[1][i-4][(j-1)/5]
            
            for j in [2, 7, 12, 17, 22]:
                for i in [8, 9, 10, 11]:
                    constrB[i, j] = constrAll[2][i-8][(j-2)/5]
                    
            for j in [3, 8, 13, 18, 23]:
                for i in [12, 13, 14, 15]:
                    constrB[i, j] = constrAll[3][i-12][(j-3)/5]
                    
            for j in [4, 9, 14, 19, 24]:
                for i in [16, 17, 18, 19]:
                    constrB[i, j] = constrAll[4][i-16][(j-4)/5]
            

            final_constr = np.vstack((np.eye(25),constrA,constrB))
            G = matrix(final_constr * -1)
            c = np.add(qTableA[next_state_A],qTableB[next_state_B]).reshape(1,25)
            c *=-1
            c = matrix(c[0])
            h = matrix(np.zeros(final_constr.shape[0]))
            A = matrix(np.ones((1,25)))
            b = matrix(1.0)

            result = solvers.lp(c,G,h,A,b)['x']

            if result !=None:
            
                qAmultiplier = np.array(qTableA[next_state_A]).flatten().reshape(25,1)
                qBmultiplier = np.array(qTableB[next_state_B]).flatten().reshape(25,1)

                va = result * qAmultiplier
                vb = result * qBmultiplier
                VA = sum(va)
                VB = sum(vb)

                if done_A == 0 and done_B == 0:
                    VA = VA  
                    VB = VB

                else:
                    VA = 0
                    VB = 0
                
                
                qTableA[cur_state_A, cur_a_A, cur_a_B] = qTableA[cur_state_A, cur_a_A, cur_a_B] + alpha * ((r_A[0] + r_B[0]) + gamma * VA - qTableA[cur_state_A, cur_a_A, cur_a_B])
                qTableB[cur_state_B, cur_a_A, cur_a_B] = qTableB[cur_state_B, cur_a_A, cur_a_B] + alpha * ((r_A[1] + r_B[1]) + gamma * VB - qTableB[cur_state_B, cur_a_A, cur_a_B])
                
                cur_state_A = next_state_A
                cur_state_B = next_state_B
                                
                done = done_A + done_B
                
                #q_error1.append(qTableA[15, 1, 4])
                q_error2.append(qTableA[71, 1, 4])

        alpha = decay(alpha)
        
        if e%1000==0:
            print("Episode number ", e, " Alpha ", alpha)
        
    return q_error, q_error1, q_error2, qTableA, qTableB


solvers.options['show_progress'] = False
solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}
game = Soccer_Game()
q_error, q_error1, q_error2, qTableA, qTableB  = CEQ(game, 0.9, 0.8, 200000)
#np.save("CEQcopy_q_a_list1", q_error1)
np.save("CEQ_q_a_list2", q_error2)
np.save("CEQ_qTableA", qTableA)
np.save("CEQ_qTableB", qTableB)
plot(np.array(q_error2), "Correlated_Q", "Correlated_Q")