# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:45:25 2020

@author: Jipeng
"""

# Improve dijkstraNewest_any grid
import sys
#Number of vertices in the graph
V = 5
# =============================================================================
# Define dijkstra to preprocess data to get paths between any two nodes in the grid
# =============================================================================
def minCost(cost, sptSet):
    # Initialize min value
    min = sys.maxsize
    # Note on 1D index
    # Given 2D index [u][v] into a V X V array
    # An equivalent 1D index can be calculated as: t= u * V + v
    # The 1D index can be converted back to 2D as: u= t/V; v = t % V
    for u in range(V):
        for v in range(V):
            if sptSet[u][v] is False and cost[u][v] <= min:
                min = cost[u][v]
                min_1D_index = u*V + v
    return min_1D_index

def dijkstraTurns(graph, turnCost, src):#-----------#
    cost = [[sys.maxsize for i in range(V)] for j in range(V)]
    # The output array cost[i][j] will hold the smallest cost from src through edge(i,j)
    sptSet = [[False for i in range(V)] for j in range(V)]
    # sptSet[i][j] will be true if edge(i,j) is included in smallest path tree or 
    # smallest cost from src to edge(i,j) is finalized
    # Initialize all costs[][] as sys.maxsize and sptSet[][] as false
    # Cost of source vertex to itself is always 0
    cost[src][src] = 0
    path = [[ [] for i in range(V)] for j in range(V) ]
    # Find shortest path for all edges
    for count in range(V * V ): # Update, it should be range(V * V) rather than range(V*V-1)
        # Pick the minimum cost edge from the set of edges not
        # yet processed. u is always equal to src * V + src in the first iteration
        min_1D_index = minCost(cost, sptSet)
        u = int(min_1D_index / V)
        v = int(min_1D_index % V)
        # Mark the picked edges as processed
        sptSet[u][v] = True
        path[u][v].append(u)
        path[u][v].append(v)
        # Update cost value of the adjacent vertices of the picked edge
        for w in range(V):
            # Update cost[v][w] only if it is not in sptSet, there is an edge from
            # v to w, and total weight of path from src to edge (v, w) through u is 
            # smaller than current value of cost [v][w]
            if (sptSet[v][w] is False) and graph[v][w]!=0 and cost[u][v] != sys.maxsize \
            and cost[u][v] + graph[v][w] + turnCost[u,v,w] < cost[v][w]:
                cost[v][w] = cost[u][v] + graph[v][w] + turnCost[u,v,w]
                path[v][w] = []
                # update path[v][w]
                for node in path[u][v]:
                    path[v][w].append(node)
    # print the constructed distance array
    printSolutionTurns(cost, path) # Further action is needed for a beautiful return
    
def printSolutionTurns(cost, path):
    print("Edge distance from Source\n")
    for i in range(V):
        for j in range(V):
            if cost[i][j] < sys.maxsize:
#                print("Distance to edge (%d,%d): %d\n",i,j,cost[i][j])
                print("Distance to edge (",i, ",",j,"): ",cost[i][j], "\n")
    
    print("Edge path from Source\n")
    for i in range(V):
        for j in range(V):
            if cost[i][j] < sys.maxsize:
                print("Path to edge (",i,",",j,"): ", path[i][j], "\n")
    
    furtherAction(cost, path)
    
def furtherAction(cost, path):
    dist = [sys.maxsize] * V
    for j in range(V):
        for i in range(V):
            if cost[i][j] < dist[j]:
                dist[j] = cost[i][j]
    minPath = [[] for i in range(V)]
    for j in range(V):
        for i in range(V):
            if cost[i][j] == dist[j]:
                minPath[j] = list(path[i][j])
    
    #-----------printSolution#
    for i in range(V):
        print("Distance to node", i,": ", dist[i],"\n")
#    for i in range(V):
#        print("Path to node", i,": ",minPath[i],"\n")
    #-----------ebd
    
    for i in range(V):
        #use dic fromkeys() method to delete overlap
        temp_list = {}.fromkeys(minPath[i]).keys()
        minPath[i] = list(temp_list)
    
    #-----------FurtherPrint
    for i in range(V):
        print("Path to node", i,": ",minPath[i], "\n")

# =============================================================================
# Define dada function
# =============================================================================
import math
import numpy as np
def create_coord_x(colNumber, rowNumber): 
    nodeNumber=colNumber * rowNumber
    coord_x=[0 for node_0 in range(nodeNumber)]
    for node in range(nodeNumber):
        coord_x[node]=node % colNumber
    return coord_x

def create_coord_y(colNumber, rowNumber): 
    nodeNumber=colNumber * rowNumber
    coord_y=[0 for node_0 in range(nodeNumber)]
    for node in range(nodeNumber):
        coord_y[node]=math.floor(node / colNumber)
    return coord_y
 
def create_D(nodesNumber, coord_x, coord_y):
    n = nodesNumber
    D = [[0 for i_i in range(n)] for j_j in range(n)]
    nodes=range(0, n)
    for i in nodes:
        for j in nodes:
            D[i][j]=np.hypot(coord_x[i]-coord_x[j],coord_y[i]-coord_y[j])                  
    return D
    
# =============================================================================
# Initialize Data
# =============================================================================
colNumber = 2
rowNumber = 2
nodesNumber = colNumber * rowNumber
coord_x=create_coord_x(colNumber,rowNumber)
coord_y=create_coord_y(colNumber,rowNumber)
D=create_D(nodesNumber, coord_x, coord_y)

# =============================================================================
# Calculate function
# =============================================================================
def distance(firstNode, secondNode,coord_x,coord_y):
    i=firstNode
    j=secondNode
    distanceValue=0
    distanceValue=np.hypot(coord_x[i]-coord_x[j],coord_y[i]-coord_y[j])
    return distanceValue

def angle(second_to_lastNode, lastNode, newNode, coord_x, coord_y):
    o=second_to_lastNode
    p=lastNode
    q=newNode
    radians_to_degrees = 180/(math.pi)
    theta_radians=0
    theta_degrees=0
    distance_o_p=distance(o,p,coord_x,coord_y)
    distance_p_q=distance(p,q,coord_x,coord_y)
    distance_o_q=distance(o,q,coord_x,coord_y)
    theta_radians=math.pi-np.arccos(round((distance_o_p**2+distance_p_q**2-distance_o_q**2)/(2*distance_o_p*distance_p_q),2))
    theta_degrees=theta_radians * radians_to_degrees
    return theta_degrees

# =============================================================================
# UB-ANC constraints
# =============================================================================
lamda = 0.1164
garma = 0.0173    

if __name__ == '__main__':
    
    V=colNumber * rowNumber
    graph = [[0 for node1 in range(V)] for node2 in range(V)]
    for firstNode in range(V):
        for secondNode in range(V):
            graph[firstNode][secondNode] = distance(firstNode, secondNode,coord_x,coord_y)
            
    #turnCost?
    three = [(i,j,k) for i in range(V) for j in range(V) for k in range(V)]
    turnCost = {(i,j,k): 0 for i,j,k in three}
    for firstNode in range(V):
        for secondNode in range(V):
            for thirdNode in range(V):
                temp = angle(firstNode, secondNode, thirdNode, coord_x, coord_y) # preprocess because of for Nan value
                if temp >= 0:
                    turnCost[firstNode, secondNode, thirdNode] = garma * temp
#                turnCost[firstNode, secondNode, thirdNode] = garma * angle(firstNode, secondNode, thirdNode, coord_x, coord_y) # non-defined turn cost cause trouble
    #turnCost = garma* angle(lastNode, newNode, departurePoint, coord_x, coord_y)

    """graph = [[0, 10, 100, 0, 0],
             [0, 0, 0, 5, 0],
             [0, 0, 0,100,0],
             [0, 0, 0, 0,10],
             [0, 0, 0, 0, 0]]
    three = [(i,j,k) for i in range(V) for j in range(V) for k in range(V)]
    turnCost = {(i,j,k): sys.maxsize for i,j,k in three}
    turnCost[1,3,4] = 10000
    turnCost[2,3,4] = 0
    turnCost[0,1,3] = 1
    turnCost[0,2,3] = 0
    turnCost[0,0,1] = 0
    turnCost[0,0,2] = 0
    """
    print("\nDijkstra with turning costs:\n")
    
    dijkstraTurns(graph, turnCost, 0)
    
#-------------------------------------------------------------------------------
# Initialize the problem data
#-------------------------------------------------------------------------------
#DEFAULT_ROLL_WIDTH = 110
"""DEFAULT_ITEMS= [(1, 20, 48),(2, 45, 35),(3, 50, 24),(4, 55, 10),(5, 75, 8)]"""
DEFAULT_LABELS= [[0,1,0],
                 [0,2,0],
                 [0,3,0]] # (1,1), (2,1) etc This should be feasible for soluti initial solution, we can develep it later
#DEFAULT_PATTERN_ITEM_FILLED=[(p, p, 1) for p in range(1,6)] # pattern1 for item1, pattern 2 for item2, etc

FIRST_GENERATION_DUALS = [1, 1, 1, 1, 0]
DEFAULT_LABELS_COST = [0,1,2]
DEFAULT_NODES = [1,2,3] #0 is excluded, as 0 is departure node
# =============================================================================
# Define master model randomly
# =============================================================================
#def masterModel(labelPool, cost, **kwargs):

 #   model = Model(name='')
#Nodes = [i for i in range(nodesNumber) if i not in obstacles and i!=departurePoint]
#node_table = [i for i in range(nodesNumber) if i not in obstacles and i!=departurePoint]

def make_EECPP_master_model(label_table, node_table, cost_table, **kwargs):
    m = Model(name='EECPP_master', **kwargs)
    m.labels = [label for label in label_table]
    m.nodes = [node for node in node_table]
    m.costs = [cost for cost in cost_table]
    nodesNumber = len(node_table) + 1 # assume departure point not in node_table
    label_number = len(label_table)

    #---variables---
    #one variable per label
    m.choose_vars = m.binary_var_list(m.labels, name="choose")
    
    # add constraints
    all_labels = m.labels
    all_nodes = m.nodes
    a = [[0 for i in range(nodesNumber - 1)] for j in range(label_number)]
    labels = [l for l in range(label_number)]
    for i in range(label_number):
        for j in range(1, nodesNumber):
            if j in all_labels[i]:
                a[i][j-1]=1
    for node in all_nodes:
        node_visit_ct = m.Sum(x[l] * a[l][node-1] for l in range(label_number)) >= 1
        node_visit_ct_name = 'ct_visit{0!s}'.format(node)
        m.node_visit_cts.append(node_visit_ct)
    m.add_constraints(m.node_visit_cts)
    
    # minimize total cost
    m.total_visiting_cost = m.sum(m.choose_vars[p] * m.costs[p] for p in range(label_number))
    m.minimize(m.total_visiting_cost)
    
    return m

def add_label_to_master_model(master_model, item_usages):
    """Adds a new label to the master model
    
    This function performs the following:
    
    1. build a new label instance from item usages(taken from sub-model, here it is dynamic programming)
    2. add it to master model
    3. update decision objects with this new label
    """
    new_label_id = max(lb.id for lb in master_model.labels) + 1
    new_label = TLabel(new_label_id, 1)
    maseter_model.labels.append(new_label)
    
    # ---add one decision variable, linked to this new label
    new_label_choose_var = master_model.binary_var(name="choose_{0}".format(new_label_id))
    
    master_model.choose_vars[new_label] = new_label_choose_var
    # update constraints
    for node, ct in zip(master_model.nodes, master_model.node_visit_cts):
        # update visit constraint by changing lhs
        ctlhs = ct.lhs
        visited = master_model.label_node_visited[new_label, node]   # what is label_node_visited
        if visited:
            ctlhs += new_label_choose_var * visited

    # update objective
    # side effect on the total traveling cost expr propgates to the objective.       
    cost_expr = master_model.total_visiting_cost
    cost_expr += new_label_choose_var * new_label.cost # this performs a side effect
    
    return master_model

def make_EECPP_label_generation_model(nodes, **kwargs):
    gen_model = Model(name = 'eecpp_generate_labels', **kwargs)
    # store data
    gen_model.nodes = nodes
    # default values
    gen_model.duals = [1] * len(nodes)
    # 1. create variables: one per node
    gen_model.use_vars = gen_model.binary_var_list(keys=nodes, name='use')
    # 2. setup constraint
    # --- sum of item usage times item sizes must be less than roll width
    gen_model.add(gen_model.dot(gen_model.use_vars, (it.size for it in items)) <= roll_width) #question
    
    
    # store dual expression for dynamic edition
    gen_model.use_dual_expr = 1 - gen_model.dot(gen_model.use_vars, gen_model.duals)
    # minimize
    gen_model.minimize(gen_model.use_dual_expr)

    return gen_model

def EECPP_solve(node_table, label_table, cost_table, **kwargs):
    verbose = kwargs.pop('verbose', True)
    master_model = make_EECPP_master_model(node_table, label_table, cost_table, **kwargs)
    # these two fields contain named tuples
    nodes = master_model.nodes
    labels = master_model.labels
    gen_model =  make_EECPP_label_generation_model(nodes, **kwargs)
    
    rc_eps = 1e-6
    obj_eps = 1e-4
    loop_count = 0
    best = 0
    curr = 1e+20
    ms = None
    
    while loop_count < 100 and abs(best - curr) >= obj_eps:
        ms = master_model.solve(**kwargs)
        loop_count += 1
        best = curr
        if not ms:
            print("{}> master model solve fails, stop".format(loop_count))
            break
        else:
            assert ms
            curr = master_model.objective_value
            if verbose:
                print('{}> new column generation iteration, #patterns={}, best={:g}, curr={:g}'
                      .format(loop_count, len(labels), best, curr))
            duals = master_model.dual_values(master_model.node_visit_cts)
            if verbose:
                print('{0}> moving duals from master to sub model:{1}'
                     .format(loop_count, list(map(lambda x: float('%0.2f' %x),duals))))
            EECPP_update_duals(gen_model,duals)
            gs = gen_model.solve(**kwargs)
            if not gs:
                print('{}> slave model fails, stop'.format(loop_count))
                break   
            
            rc_cost = gen_model.objective_value
            if rc_cost <= -rc_eps:
                if verbose:
                      print('{}> runs with obj = {:g}'.format(loop_count, rc_cost))
            else:
                if verbose:
                    print('{}> pattern-generator model stops, obj = {:g}'.format(loop_count, rc_cost)) # no question
                break
            
            use_values=gen_model.solution.get_values(gen_model.use_vars) # mark as it need to be continue finishing
            if verbose:
                print('{}> add new label to master data: {}'.format(loop_count, str(use_values))) # not sure
            # make a new label with use values
            if not (loop_count <100 and abs(best - curr) >= obj_eps):
                print('* terminating: best- curr = {:g}'.format(abs(best-curr)))
                break
            add_label_to_master_model(master_model, use_values)
    if ms:
        if verbose:
            print('\n* EECPP column generation terminates, best={:g}, #loops={}'.format(curr, loop_count))
            EECPP_print_solution(master_model)
        return ms
    else:
        print('!!!!  EECPP column generation fails  !!!!!')
        return None            
                
def EECPP_print_solution(EECPP_model):
    labels = EECPP_model.labels
    choose_var_values = {p: EECPP_model.choose_vars[p].solution_value for p in labels}
#    pattern_item_filled = cutstock_model.pattern_item_filled
    print("/ Nb of chooses / Label")
    print("/ {} /".format("-" * 70)) 
    for p in labels:
        if choose_var_values[p] >= 1e-3:
            label_detail = {b.id: pattern_item_filled[a,b] for a,b in pattern_item_filled if
                              a==p}
            print(
                "| {:<10g} | {!s:9} |".format(choose_var_values[p],
                                                       p)
    print("| {} |".format("-" * 70))    

    


def EECPP_update_duals(gen_model, new_duals):
    # update the duals array and the duals expression...
    # edition is propagated to the objective of the model
    gmodel.duals=new_duals
    use_vars=gmodel.use_vars
    assert len(new_duals) == len(use_vars)
    updated_used = [(use, -new_duals[u]) for u, use in enumerate(use_vars)]    
    # this modification is notified to the objective.
    gmodel.use_dual_expr.set_coefficients(updated_used)  
    return gmodel

    


def EECPP_solve_default(**kwargs):
    return EECPP_solve(DEFAULT_NODES, DEFAULT_LABELS, DEFAULT_LABELS_COST, **kwargs)


#def EECPP_save_as_json:
    
# =============================================================================
# make_EECPP_label_generation
# =============================================================================
    
# Initialize programe data
    
import time
import numpy as np
import math
number_of_criteria_based_on_adjacent_nodes=0
number_of_criterion1_eliminated=0
number_of_optimalityPrinciple_eliminated=0
colNumber = 2
rowNumber = 2
lamda = 0.1164
garma = 0.0173

def obtain_all_qualified_labels(D,coord_x, coord_y, nodesNumber, Battery_capacity_constraint, departurePoint, obstacles):
    global lamda
    global garma
    start_time = time.time()
    totalCost = 0
    lastNode=[]
    second_to_lastNode=[]
    iteration = nodesNumber
    sets=[]
    all_qualified_labels=[]
    nodes=[node for node in range(0,nodesNumber) if node!= departurePoint and node not in obstacles] # new node to be added
    for newNode in nodes:
        totalCost = 0
        feasible_set=[departurePoint]
        lastNode=departurePoint
        second_to_lastNode=departurePoint # This is special case for k=1 iteration
        pass0 = Elimination_criteria_based_on_obstacle(obstacles, lastNode, newNode)
        if pass0 is True:
            pass1 = Battery_capacity_limit_check(second_to_lastNode, lastNode, newNode, coord_x, coord_y, D , totalCost, Battery_capacity_constraint )
            if pass1 is True:
                feasible_set.append(newNode)
                feasible_label = list(feasible_set)
                go=lastNode
                to=newNode
                turnCost=0
                distanceCost=lamda*D[go][to]
                totalCost=turnCost+distanceCost
                label=[totalCost]
                feasible_set.append(label)
                sets.append(feasible_set)
                pass0 = Elimination_criteria_based_on_obstacle(obstacles, newNode, departurePoint)
                if pass0 is True:
                    permission = Battery_capacity_limit_check(lastNode, newNode, departurePoint, coord_x, coord_y, D, totalCost, Battery_capacity_constraint)
                    if permission is True:
                        turnCost = garma* angle(lastNode, newNode, departurePoint, coord_x, coord_y)
                        distanceCost=lamda* D[newNode][departurePoint]
                        totalCost=totalCost+turnCost + distanceCost
                        label=[totalCost]
                        feasible_label.append(departurePoint)
                        feasible_label.append(label)
                        all_qualified_labels.append(feasible_label)
               
    # node in start end
    label=[]
    # update thanks to new requirement: ending point must be exactly same with leaving point
    # nodeNumber -2(previously -3) nodes is added here
    for k in range(1, iteration-1):
        flag = 0 # flag work as flag, once at the iteration, no new node is eligible to be added, break(way: return) the loop to save time
        sets_copy=list(sets)
        sets=[]
        for S_0 in sets_copy:
            second_to_lastNode = S_0[-3]
            lastNode = S_0[-2]
            S=S_0.copy()
            S.pop()
            for newNode in nodes: # nodes replace nodesAndArrival to improve efficiency
                label=[]
                totalCost=0
                feasible_set = list(S_0) # Make a shallow copy # It is not good enough now because S includes label at last posion
                label=feasible_set.pop() # Added thanks to the previous line
                totalCost=label[0]
                pass0 = Elimination_criteria_based_on_obstacle(obstacles, lastNode, newNode)
                if pass0 is True:
                    pass1 = Battery_capacity_limit_check(second_to_lastNode, lastNode, newNode, coord_x, coord_y, D , totalCost, Battery_capacity_constraint )
                    if pass1 is False:
                        pass
                    else:
                        pass2 = Elimination_criteria_based_on_the_set_S_only(S, newNode)
                        if pass2 is False:
                            pass
                        else:
                            flag = 1
                            feasible_set.append(newNode)
                            feasible_label = list(feasible_set)
                            go=lastNode
                            to=newNode
                            turnCost=garma* angle(second_to_lastNode, go, to, coord_x, coord_y)
                            distanceCost=lamda* D[go][to]
                            totalCost=totalCost+turnCost + distanceCost
                            label=[totalCost]
                            feasible_set.append(label)
                            sets.append(feasible_set)
        sets=optimalityPrinciple(sets)
        for feasible_set in sets:
            lastNode = feasible_set[-2]
            second_to_lastNode = feasible_set[-3]
            newNode = departurePoint
            totalCost = feasible_set[-1][0]
            pass0 = Elimination_criteria_based_on_obstacle(obstacles, lastNode, newNode)
            if pass0 is True:
                permission = Battery_capacity_limit_check(second_to_lastNode, lastNode, newNode, coord_x, coord_y, D, totalCost, Battery_capacity_constraint)
                if permission is True:
                    turnCost = garma * angle(second_to_lastNode, lastNode, newNode, coord_x, coord_y)
                    distanceCost = lamda * D[lastNode][newNode]
                    totalCost = feasible_set[-1][0] + turnCost + distanceCost
                    label = [totalCost]
                    feasible_set_copy = list(feasible_set[:-1])
                    feasible_set_copy.append(newNode)
                    feasible_set_copy.append(label)
                    all_qualified_labels.append(feasible_set_copy)
        if flag == 0:
            return all_qualified_labels
        # Time limit to 3600s
        currentTime = time.time()
        differentTime = currentTime - start_time
        if differentTime >= 3600:
            return all_qualified_labels
        # end
    return all_qualified_labels

def Elimination_criteria_based_on_obstacle(obstacles, lastNode, newNode):
    global colNumber
    sideLength = 1
    flag=1 # initialize flag=1
    for obstacle in obstacles:
        x3 = obstacle % colNumber
        y3 = obstacle // colNumber
        r_xmin = x3 - sideLength/2
        r_xmax = x3 + sideLength/2
        r_ymin = y3 - sideLength/2
        r_ymax = y3 + sideLength/2
        if RSIntersection(r_xmin, r_xmax, r_ymin, r_ymax, lastNode, newNode) == 1:
            flag = 0 #flag is variable to replace directly return False of True because we need to test all obstacles
        else:
            flag = 1
        if flag == 0:#once flag change to 0, means one obstacle is hit, thus new node can not be added
            return False
    return True# This means all obstacles are not hit, new node can be considered to be added
"""
 * if [x1, x2] abd [x3, x4] (x4 maybe smaller than x3) has interrection or has not，if yes: return 1， if no: return 0
"""
def IntervalOverlap(x1, x2, x3, x4):
    t = 0    
    if x3 > x4:
       t = x3
       x3 = x4	   
       x4 = t
    if x3 >= x2 or x4 <= x1:
        return 0
    else:
        return 1
    
"""
 * judge rectangular r and line segment AB has intersection or not，if yes: return 1，if no: return 0
"""
def RSIntersection(r_xmin,r_xmax,r_ymin, r_ymax, nodeA, nodeB):
    global colNumber
    global rowNumber
    A_x = nodeA % colNumber
    A_y = nodeA // colNumber
    B_x = nodeB % colNumber
    B_y = nodeB // colNumber
    if (A_y == B_y):# line segment is parallet to  x axix
        if A_y <= r_ymax and A_y >= r_ymin:
            return IntervalOverlap(r_xmin, r_xmax, A_x,B_x)
        else:
            return 0

    # exange node A and node B, make B's y value bigger
    # Exchange node A and node B, let B's y value is bigger
    t = 0
    if A_y > B_y:
       t = A_y
       A_y = B_y
       B_y = t
       t= A_x
       A_x = B_x
       B_x=t
	
    # In line segment AB, to find point C and D
    # Two points secure a line: (x-x1)/(x2-x1)=(y-y1)/(y2-y1)
    k = (B_x - A_x)/(B_y - A_y)
    if A_y < r_ymin:
       D_y = r_ymin
       D_x = k*(D_y - A_y) + A_x
    else:
       D_y=A_y
       D_x=A_x
    if B_y > r_ymax:
       C_y = r_ymax
       C_x = k*(C_y-A_y) + A_x
    else:
       C_y = B_y
       C_x = B_x
    if C_y >= D_y: # y axis has overlap
       return IntervalOverlap(r_xmin, r_xmax,D_x, C_x)
    else:
       return 0
#end

# As the algorithm is based on dynamic programming, it is useful to limit the number of states created
# For a state (S,i) to which node j is to be added, the frist four elimination tests described for state(SU{j},j) are independent of
# the terminal node i
# while the next four tests depend on the labels in H(S,i)
# To simplify the execution of the tests, information on the states is stored in  a three level structure
# At the first level, we have the set S and its load y(S)
# At the second level,for the given set S, we have the terminal nodes i belongs to S used to form the states (S,i)
# At the third level, for a state (S,i), the reduced set of lables H(S,i) is stored.
def Battery_capacity_limit_check(second_to_lastNode, lastNode, newNode, coord_x, coord_y, D , totalCost, Battery_capacity_constraint ):
    global lamda
    global garma
    futureCost=0
    go=lastNode
    to=newNode
    turnCost=garma * angle(second_to_lastNode, go, to, coord_x, coord_y)
    #judge turnCost nan or not nan
    flag = math.isnan(turnCost)
    #end
    if flag == True:
        turnCost=0
    else:
        turnCost=turnCost
    distanceCost=lamda * D[go][to]
    totalCost=totalCost+turnCost + distanceCost
    futureCost = totalCost
    if futureCost > Battery_capacity_constraint:
        return False
    return True

def Elimination_criteria_based_on_the_set_S_only(S,newNode):
    global number_of_criterion1_eliminated
    j = newNode
    S = S # we have the set S
    # criterion 1
    if j in S: # criterion #1, new node should not have been visited
        number_of_criterion1_eliminated+=1
        return False
    # criterion 1 end
    return True 

def optimalityPrinciple(sets): # it needs to change because sets each set has a label as last element, this is different from before
    global number_of_optimalityPrinciple_eliminated
    the_nodes_of_R=[]
    sets_copy=sets.copy()
    R = []
    for R in sets:
        the_nodes_of_R = set(R[:-1]) #[:-1] is added now because we don't want label as node
        for R_2 in sets_copy:
            if R_2!=R:
                if set(R_2[:-1]) == the_nodes_of_R:
                    if R_2[-2] == R[-2]:
                        if R_2[-3] == R[-3]:
                            if R_2[-1][0]<=R[-1][0]:
                                number_of_optimalityPrinciple_eliminated+=1
                                sets_copy.remove(R) # The label is eliminated if there exists another label with both a lesser time and a lesser cost
                                break
    return sets_copy

def distance(firstNode, secondNode,coord_x,coord_y):
    i=firstNode
    j=secondNode
    distanceValue=0
    distanceValue=np.hypot(coord_x[i]-coord_x[j],coord_y[i]-coord_y[j])
    return distanceValue

def angle(second_to_lastNode, lastNode, newNode, coord_x, coord_y):
    o=second_to_lastNode
    p=lastNode
    q=newNode
    radians_to_degrees = 180/(math.pi)
    theta_radians=0
    theta_degrees=0
    distance_o_p=distance(o,p,coord_x,coord_y)
    distance_p_q=distance(p,q,coord_x,coord_y)
    distance_o_q=distance(o,q,coord_x,coord_y)
    theta_radians=math.pi-np.arccos(round((distance_o_p**2+distance_p_q**2-distance_o_q**2)/(2*distance_o_p*distance_p_q),2))
    theta_degrees=theta_radians * radians_to_degrees
    return theta_degrees
    

#-------------------------------------------------------------------------------
# Solve the model and display the result
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    s = EECPP_solve_default()
    assert abs(s.objective_value - 46.25) <= 0.1
     Save the solution as "solution.json" program output.
    with get_environment().get_output_stream("EECPP_solution.json") as fp:
        EECPP_save_as_json(s.model,fp)






    