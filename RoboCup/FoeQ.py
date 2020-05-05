import numpy as np
from Environment import Soccer_Game, Player
from random import choice
import random
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

def FoeQ(game, gamma, alpha, episodes):

	q_table = np.ones((112, 5, 5))
	q_error = []
	action_space = game.action_space
	naction = len(action_space)
	qInd = qTableInd()
	
	for e in range(episodes):
		game.reset()
		done = 0
		total_rewardA = 0
		total_rewardB = 0

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


			next_pos_A = game.playerA.x * 4 + game.playerA.y
			next_pos_B = game.playerB.x * 4 + game.playerB.y

			ball = game.playerB.ball

			next_state_A = qInd[str(ball) + str(next_pos_A) + str(next_pos_B)]
			next_state_B = qInd[str(ball) + str(next_pos_A) + str(next_pos_B)]

			qT = q_table[next_state_A].T
			constraint = np.vstack((qT,np.eye(5)))

			x_constr = np.zeros(10)
			x_constr[:5] = -1
			x_constr[5:]=0
			x_constr=x_constr.reshape((constraint.shape[0],1))
			Allconstr = np.hstack((x_constr,constraint))
			g = Allconstr * -1
			G = matrix(g)
			h = matrix(np.zeros(g.shape[0]))
			A = matrix([[0],[1.0],[1.0],[1.0],[1.0],[1.0]])
			b = matrix(1.0)
			c = np.zeros(6)
			c[0] = -1
			c = matrix(c)
			result = solvers.lp(c,G,h,A,b)['x']

			if done_A != 0 or done_B != 0:
				V = 0
			else:
				V = result[0]

			q_table[cur_state_A, cur_a_A, cur_a_B] = (1-alpha) * q_table[cur_state_A, cur_a_A, cur_a_B] + alpha * ((1-gamma)*(r_A[0] + r_B[0]) + gamma * V)

			cur_state_A = next_state_A
			cur_state_B = next_state_B

			done = done_A + done_B
			
			q_error.append(q_table[71, 1, 4])

		alpha = decay(alpha)

		if e%1000==0:
		    print("Episode number ", e, " Alpha ", alpha)

	return q_error, q_table


solvers.options['show_progress'] = False
solvers.options['glpk'] = {'LPX_K_MSGLEV': 0, 'msg_lev': "GLP_MSG_OFF"}
game = Soccer_Game()
q_a_list, q_table = FoeQ(game, 0.9, 0.8, 200000)
#q_a_list, q_table = FoeQ(game, 0.9, 0.8, 20000)
np.save("FoeQstepcopy_q_a_list", q_a_list)
np.save("FoeQstepcopy_qTable", q_table)

plot(np.array(q_a_list), "Foe-Q", "FoeQstepcopy")


