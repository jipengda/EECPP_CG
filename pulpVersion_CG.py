# We use label to replace path, it is difference from CG 3.12.py
# Labels replace Paths

#import csv
import pulp
import time
import Data
import numpy as np



colNumber = 1
rowNumber = 1
NumberOfNodes = colNumber * rowNumber

k = 10 # number of drones

BigM = 50

#----------------------------------------------------------------
# DATA
#----------------------------------------------------------------
colNumber = 3
rowNumber = 3
C = 6.0
lamda = 0.1164
garma = 0.0173

coord_x = Data.create_coord_x(colNumber, rowNumber)
coord_y = Data.create_coord_y(colNumber, rowNumber)
NumberOfNodes = colNumber * rowNumber
Points = range(1, NumberOfNodes+1)
Nodes = range(0, NumberOfNodes+2)
c = {(i,j):0 for i in Nodes for j in Nodes}
distance = {(i,j):0 for i in Nodes for j in Nodes}
for i in Nodes:
	for j in Nodes:
        distanceValue = np.hypot(coord_x[i]-coord_x[j], coord_y[i]-coord_y[j])
        distance[(i,j)]=distanceValue
        distance_cost = lamda * distanceValue
        c[(i,j)] = distance_cost

#3x3 turning cost is 0
label_table = [[0, 1, 0, [0.2328]],
               [0, 2, 0, [0.4656]],
               [0, 3, 0, [0.2328]],
               [0, 4, 0, [0.3292]],
               [0, 5, 0, [0.5206]],
               [0, 6, 0, [0.4656]],
               [0, 7, 0, [0.5206]],
               [0, 8, 0, [0.6585]]]

#----------------------------------------------------------------
# MASTER PROBLEM MIP MODELING
#----------------------------------------------------------------
label_number = len(label_table)
MASEECPP  = pulp.LpProblem("Master Problem EECPP", pulp.LpMinimize)
MDecisionVar = []

for label in Labels:
	MDecisionVar.append(pulp.LpVariable("Label" + str(label), 0, 1, pulp.LpContinuous))	


Obj = pulp.LpConstraintVar(name = "OBJ",e= sum(cost[i] * MDecisionVar[i] for i in range(5) ))
MASEECPP.setObjective(Obj)


# this has somthing wrong
# need to confirmed later
Node = []
for label in Labels:
	Node.append(pulp.LpConstraintVar(name = "Nodes"+str(label), sense=pulp.LpConstraintEQ, rhs=1,e=MDecisionVar[label]))
	MASEECPP += Node[label]
print(Node)

#------------------------------------------------------------------
# SUBPROBLEM MIP MODELING
#------------------------------------------------------------------


SUBEECPP = pulp.LpProblem("Sub Problem EECPP", pulp.LpMinimize) 
ConstantCost = pulp.LpVariable("ConstantCost", 1 , 1) # 1 is lowever bound, the next 1 is upper bound
DecisionVar = pulp.LpVariable.dicts("Assign", (Nodes, Nodes), 0, 1, pulp.LpBinary) # Binary
ResultVar = pulp.LpVariable.dicts("Make",(Nodes, Nodes, Nodes), 0, 1, pulp.LpBinary)


SUBEECPP += pulp.lpSum([DecisionVar[0][node] for node in 
					  Nodes if node!= 0]) == 1 #(25)





SUBEECPP += DecisionVar[0][0] != 1, "None" #(26)




for point in Points:
	SUBEECPP += pulp.lpSum([DecisionVar[node][point] - DecisionVar[point][node] for node in
						   Nodes if node!=point]) == 0, "forTrip"+str(point)+'k' #(31)




for first in Nodes:
	for second in Nodes:
		for thirde in Nodes:
			SUBEECPP += ResultVar[first][second][third] >= DecisionVar[first][second] + DecisionVar[second][third] - 1, 
					"Convert Non-Linear 1"+str(first)+str(second)+str(third) #(26)
			SUBEECPP += ResultVar[first][second][third] <= DecisionVar[first][second], "Convert Non-Linear 2"+str(first)+str(second)+str(third)#(27)
			SUBEECPP += ResultVar[first][second][third] <= DecisionVar[second][third], "Convert Non-Linear 3"+str(first)+str(second)+str(third)#(28)




#(29) battery capacity constraint
SUBEECPP += pulp.lpSum( c[go][to] * DecisionVar[go][to] for go in Nodes for to in Nodes) <= C




#(30) subtour elimination constraint





tic = time.time()


#---------------------------------------------------------------
#COLUMN GENERATION ITERATIONS
#---------------------------------------------------------------


i = 0
while True:
	MASEECPP.write(str(i) +"Master.lp")
	MASEECPP.solve()
	price = {}
	price.clear()

	for label in range(0, NumberOfNodes + 1):
		price[label] = float(MASEECPP.constraints["Node"+str(label)].pi)

	print("Dual values: ", price)
	print()

	SUBEECPP += sum([ConstantCost-[price[From] * DecisionVar[From][To] for From in Nodes for To in Nodes
		   if not (From == To or From == NumberOfNodes + 1)]])
	# Objective # i DOUBT this is wrong

	SUBEECPP.solve()

	SUBEECPP.writeLP(str(i)+"SUBEECPP.lp")

	if(pulp.value(SUBEECPP.objective) > -1) or i == 50:
		break

	expression = Obj # uncertain about the meaning of expression here
	for node1 in Nodes:
		for node2 in Nodes:
            if DecisionVar[node1][node2].value() == 1.0:
                print("Point:"+ str(node1)+ " to Point:" + str(node2))
                if not (node1 == NumberOfPoints+1):
                    expression += Customer[node1]			
    print()
    print("Master LP problem objective value: ", pulp.value(MASEECPP.objective))
    print("Subproblem Objective value: ", pulp.value(SUBEECPP.objective))
    print()

    # print(expression)
    # print()

    MDecisionVar.append(pulp.LpVariable("Label"+str(len(Labels)),0,1,pulp.LpContinuous,e=expression))

    Labels.append(len(Labels)) # add new column

    print("************************************************************")
    print("Column Generation Iteration: ", i)
    print()
    i+=1

print("************************************************************")
print("****** Column Generation Done. Setting all variables to integers in the Master Problem and solving it... ******")
print()
for variables in MASEECPP.variables():
	variables.cat = pulp.LpInteger # variables.cat is what

MASEECPP.writeLP("MasterInteger.lp")

MASEECPP.solve()

toc = time.time()


for label in Labels:
	if MDecisionVar[label].value() == 1.0:
		print("Label selected is: ", str(label))

print()
