# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
try setting turning cost to 0
"""

from docplex.mp.model import Model
import Data
import numpy as np
import math
import matplotlib.pyplot as plt
import time



colNumber=3
rowNumber=3

coord_x = Data.create_coord_x(colNumber, rowNumber)
coord_y = Data.create_coord_y(colNumber, rowNumber)

# 3x3 turning cost is not 0
#label_table = [[0, 1, 0, [3.3468]], 
#               [0, 2, 0, [3.5796]],
#               [0, 3, 0, [3.3468]],
#               [0, 4, 0, [3.4432]],
#	           [0, 5, 0, [3.6346]],
#               [0, 6, 0, [3.5796]],
#               [0, 7, 0, [3.6346]],
#               [0, 8, 0, [3.7725]]]

#3x3 turning cost is 0
label_table = [[0, 1, 0, [0.2328]],
               [0, 2, 0, [0.4656]],
               [0, 3, 0, [0.2328]],
               [0, 4, 0, [0.3292]],
               [0, 5, 0, [0.5206]],
               [0, 6, 0, [0.4656]],
               [0, 7, 0, [0.5206]],
               [0, 8, 0, [0.6585]]]

             
def make_eecpp_master_model(label_table, colNumber, rowNumber, **kwargs):
    label_number = len(label_table)
    nodesNumber= colNumber * rowNumber
    departurePoint = 0
    obstacles = []
    ##
    C=[0] * label_number
    for l in range(label_number):
        C[l] = label_table[l][-1][0]

    # column vector set
    a = [[0 for i in range(nodesNumber)] for j in range(label_number)]
    labels = [l for l in range(label_number)]
    for l in range(label_number):
        for j in range(0, nodesNumber):
            if j in label_table[l]:
                a[l][j] = 1
    #model<->m
    m = Model(name='EECPP_master')
    m.labels = labels
    m.label_table = label_table
    m.label_number = label_number
    m.nodes = [i for i in range(nodesNumber)]
    m.x = m.continuous_var_dict(m.labels,lb=0, ub=1, name="visit") # determine if the label is selected(x=1) or not(x=0)
    # minimize total cost
    m.visiting_cost = m.sum( (C[l] * m.x[l]) for l in labels )
    m.minimize(m.visiting_cost)
    

    m.node_visit_cts=[]
    #
    ##########
    for node in m.nodes:
        if node in obstacles:
            node_visit_ct = m.sum(m.x[l] * a[l][node] for l in labels) >= 0
        else:
            node_visit_ct = m.sum(m.x[l] * a[l][node] for l in labels) >= 1
        node_visit_ct_name = 'ct_visit{0!s}'.format(node)
        m.node_visit_cts.append(node_visit_ct)
    m.add_constraints(m.node_visit_cts)
    return m    
def make_eecpp_generation_model(colNumber, rowNumber, coord_x, coord_y,**kwargs):
    nodesNumber = colNumber * rowNumber
    duals = [0] * nodesNumber
    pi = duals
    departurePoint = 0

    #arc_three_nodes<->arcs
    #c=6.0 battery constraint is 6.0
    
    C = 5.0
    lamda = 0.1164
    garma = 0.0173
    garma = 0 # turning cost is 0
    zero_turn = garma * 180
    C = C + zero_turn
    control_inf = C/lamda+1
    radians_to_degrees = 180/(math.pi)
    distanceValue=0
    theta_radians=0
    theta_degrees=0
    obstacles = []
    Nodes = [i for i in range(nodesNumber) if i not in obstacles and i!=departurePoint]
    # Except departurePoint( arrival point is same as departurePoint) and obstacles
    NodesAndDeparturePoint = Nodes + [departurePoint]
    AllNodes = NodesAndDeparturePoint + obstacles
    edges = [(i,j) for i in NodesAndDeparturePoint for j in NodesAndDeparturePoint]
    arcs = [(i,j,k) for i in NodesAndDeparturePoint for j in NodesAndDeparturePoint for k in NodesAndDeparturePoint]
    distance={(i,j):0 for i,j in edges}
    
    ############
    
    ############
    """
    Need add something to deal with obstacles
    if line segment of i&j interacts with obstacles, c_ij is infinity(math.inf).
    if line segment of i&j does not interact with obstacles, c_ij is european distance.
    """

    c = {(i,j):0 for i,j in edges}
    q = {(i,j,k):0 for i,j,k in arcs}
    distance={(i,j):0 for i,j in edges}
    for i,j in edges:
        distanceValue = np.hypot(coord_x[i]-coord_x[j], coord_y[i]-coord_y[j])
        distance[(i,j)]=distanceValue
        distance_cost = lamda * distanceValue
        c[(i,j)] = distance_cost
    
    for o,p in edges:
        View = check_obstacle(obstacles, o, p, colNumber, rowNumber)
        if View == 0:
            c[(o,p)] = math.inf
        else:
            pass
    turning_cost = 7
    for i,j,k in arcs:
        theta_radians=math.pi-np.arccos(round((distance[i,j]**2+distance[j,k]**2-distance[i,k]**2)/(2*distance[i,j]*distance[j,k]),2))
        theta_degrees=theta_radians*radians_to_degrees
        turning_cost=garma*theta_degrees
        q[(i,j,k)]=turning_cost
        a=math.isnan(turning_cost)
        if a is True:
            turning_cost=0
        else:
            pass
        q[(i,j,k)]=turning_cost
    

    # An arc flow model for the basic EECPP
    gen_model = Model("eecpp_generate_labels")
    gen_model.edges = edges
    gen_model.duals = [0] * nodesNumber
    gen_model.x = gen_model.binary_var_dict(edges, name = 'X') # flow variables, 1 if the agent goes directly from node i to node j
    gen_model.I = gen_model.binary_var_dict(arcs, name = "I") # use I in order to linearly replace x (i,j) x x (j,k)
    d = gen_model.continuous_var_list(AllNodes, name = "D") # d is a dummy variable associated with node i for subtour elimination

    ##### set 4 expr expressions to simplify expressions later
    gen_model.expr_1 = gen_model.sum( c[(i,j)] * gen_model.x[(i,j)] for i,j in edges)
    gen_model.expr_2 = gen_model.sum( gen_model.duals[i] * gen_model.x[(i,j)] for i,j in edges)
    # gen_model.expr_3 has mistake
    gen_model.expr_3 = gen_model.sum((q[(i,j,k)]* gen_model.I[(i,j,k)]) for i,j,k in arcs)
    # turn cost calculation is wrong
    gen_model.expr_4 = gen_model.sum( (q[(i,0,k)] * gen_model.I[(i,0,k)]) for i,k in edges)
    #####
    
    gen_model.expr = gen_model.expr_1 - gen_model.expr_2 + gen_model.expr_3 - gen_model.expr_4
    gen_model.labelCost = gen_model.expr + gen_model.expr_2
#    gen_model.energyCost = gen_model.expr_1 + gen_model.expr_3
    gen_model.energyCost = gen_model.expr_1 # if turning cost is 0 , we dont need gen_model.expr_3.
    
    gen_model.minimize(gen_model.expr) #(24)
    
    # 2. setup constraint
    gen_model.add_constraint(gen_model.sum( gen_model.x[(0,j)] for j in Nodes) == 1, ctname = None) #(25)
    gen_model.add_constraint(gen_model.x[(0,0)] !=1, ctname= None) # (26)


    #once a drone visits a node, it also departs from the same node. Each node can and can only be visited only once.
    for j in NodesAndDeparturePoint:
        gen_model.add_constraint(gen_model.sum( gen_model.x[(i,j)] for i in NodesAndDeparturePoint)==
                                 gen_model.sum( gen_model.x[(j,k)] for k in NodesAndDeparturePoint), ctname = None) #(31)
        
    for i,j,k in arcs:
        gen_model.add_constraint( gen_model.I[(i,j,k)] >= gen_model.x[(i,j)] + gen_model.x[(j,k)] - 1, ctname = None ) #(26)
        gen_model.add_constraint( gen_model.I[(i,j,k)] <= gen_model.x[(i,j)], ctname = None ) #(27)
        gen_model.add_constraint( gen_model.I[(i,j,k)] <= gen_model.x[(j,k)], ctname = None ) #(28)

    #(29) battery capacity constraint
    gen_model.add_constraint( gen_model.energyCost <= C, ctname = None )
    #(30) subtour elimination constraint
    for i,j in edges:    
        if j!=departurePoint:
            gen_model.add_indicator( gen_model.x[(i,j)],d[i] + 1 == d[j],name=None)
    
#    gen_model.add_constraint(gen_model.expr<=0)
    
    return gen_model

def eecpp_update_duals(gen_model, new_duals):
    gen_model.duals = new_duals
    gen_model.expr_2 = gen_model.sum( gen_model.duals[i] * gen_model.x[(i,j)] for i,j in gen_model.edges)
    gen_model.expr = gen_model.expr_1 - gen_model.expr_2 + gen_model.expr_3 - gen_model.expr_4
    gen_model.labelCost= gen_model.expr + gen_model.expr_2
    gen_model.minimize(gen_model.expr)
    return gen_model

def add_pattern_to_master_model(master_model, colNumber, rowNumber, x_values, label_cost, outF):
    
    nodesNumber = colNumber * rowNumber
    departurePoint = 0
    obstacles = []
    Nodes = [i for i in range(nodesNumber) if i not in obstacles and i!=departurePoint]
    NodesAndDeparturePoint = Nodes + [departurePoint]
    edges = [(i,j) for i in NodesAndDeparturePoint for j in NodesAndDeparturePoint]
    draw_edges =[i for i in edges if x_values[i].solution_value>0.9]
    gen_model_label=[0]    
    sad=0
    iteration = len(draw_edges)
    x=x_values
    for i in range(iteration):
        for happy in NodesAndDeparturePoint:
            if x[(sad, happy)].solution_value > 0.9:
                gen_model_label.append(happy)
                sad = happy
                break
    print("Column Generation Label and its cost:")
    LabelAndItsCost="Column Generation Label and its cost:" +"\n" + str(gen_model_label) + "\n" + str(label_cost)
    outF.write(LabelAndItsCost)
    outF.write("\n")
#    print(gen_model_label)

    #
    # last difficult about nameing new_lable and giving the right ID
    new_label=gen_model_label
    new_labelCost = label_cost
    print(new_label)
    print(new_labelCost)
    new_label.append([new_labelCost])
    max_label_table_id = len(master_model.label_table)
    master_model.label_table.append(new_label)
    new_label_id = max_label_table_id
    master_model.labels.append(new_label_id)
    new_label_x = master_model.continuous_var(lb=0, ub=1, name = 'visit_{0}'.format(new_label_id))
    master_model.x[new_label_id] = new_label_x
    
    
    # update constraints
    for node, ct in zip(master_model.nodes, master_model.node_visit_cts):
        ctlhs = ct.lhs
        if node in new_label:
            ctlhs = ctlhs + new_label_x * 1
            ct.lhs = ctlhs
            
    cost_expr = master_model.visiting_cost
    cost_expr = cost_expr + new_label_x * new_labelCost
    master_model.visiting_cost = cost_expr
    master_model.minimize(master_model.visiting_cost)
    return master_model

def eecpp_solve(colNumber, rowNumber, label_table, coord_x, coord_y, **kwargs):
    master_model = make_eecpp_master_model(label_table, colNumber, rowNumber, **kwargs)
    
    gen_model = make_eecpp_generation_model(colNumber, rowNumber, coord_x, coord_y, **kwargs)
    
    tic = time.time()
    
    obj_eps = 1e-4
    rc_eps = 1e-6
    loop_count = 0
    best=0
    curr=1e+20
    SplitLines="*************************************************************"
    ms = None
    # write column generation results to Loop count.txt
    outF = open("Find Details.txt", "w")
    
#    while loop_count < 100 and abs(best - curr) >= obj_eps:
#Staring right after the fourth # is where while begins
####################    
    # list 1 is used to store loop_count
    # list 2 is used to store sum of master_model.x[i].solution_value
    list1=[]
    list2=[]
#    while loop_count < 100 and abs(best - curr) >= obj_eps:
    while loop_count < 100:
        list1.append(loop_count)
        ms = master_model.solve(log_output=True)
        internalMonitor = 0
        for i in master_model.labels:
            internalMonitor = internalMonitor + (master_model.x[i]).solution_value
        list2.append(internalMonitor)
        loop_count += 1
        best = curr
        curr = master_model.objective_value
        duals = master_model.dual_values(master_model.node_visit_cts)
        eecpp_update_duals(gen_model, duals)
        gs = gen_model.solve(log_output = True)
        if not gs:
            print('{}> slave model fails, stop'.format(loop_count))
            break
        rc_cost = gen_model.objective_value
        if( rc_cost >= -rc_eps): # use -rc_eps instead of 0 to protect from looping infinitely
            print("break because of rc_cost >= -rc_eps!!!")
            print("And the rc_cost is {}!!!".format(rc_cost))
            break
        label_cost = gen_model.labelCost.solution_value

        print("*************************************************************")
        print("Column Generation Iteration: ", loop_count)
        CGI_count="Column Generation Iteration: " + str(loop_count)
        outF.write(SplitLines)
        outF.write("\n")
        outF.write(CGI_count)
        outF.write("\n")
        MM="Master Model Solution and its cost: "
        outF.write(MM)
        outF.write("\n")
        for i in master_model.labels:
            if (master_model.x[i]).solution_value > 0.9:
                ms_label ="master model solution labels: " +  str(master_model.label_table[i])
                outF.write(ms_label)
                outF.write("\n")
        ms_objective = "master model objective: " + str(master_model.objective_value)
        outF.write(ms_objective)
        outF.write("\n")

        Dual_values="Dual values:"+str(duals)
        outF.write(Dual_values)
        outF.write("\n")
        rc_cost="rc_cost:"+str(rc_cost)
        outF.write(rc_cost)
        outF.write("\n")
        master_model=add_pattern_to_master_model(master_model, colNumber, rowNumber, gen_model.x, label_cost, outF)
        
# put it in the loop every time we get a optimal solution for master model
# check optimal solution rather than intermediate solution
#Staring right after the fourth # is where while begins
####

# print out the solutiino of master_model before judge the sum is integer or not
#
    eecpp_print_solution(master_model, outF)    
    plt.plot(list1, list2)
    plt.title("sum VS loop_count")
    plt.xlabel("loop_count")
    plt.ylabel("sum")
    m = 0  
    for i in master_model.labels:
        m = m + (master_model.x[i]).solution_value
    if m.is_integer() is False:
        next_integer = math.ceil(m)
        labels = master_model.labels
        master_model.global_cut_ct = master_model.sum(master_model.x[l] for l in labels) >= next_integer
        (master_model.global_cut_ct_name) = 'global_cut_ct_1' # the name may be overlapped , which will cause error.
        master_model.add_constraint(master_model.global_cut_ct)
            
        alpha = (master_model.global_cut_ct).dual_value

            #alpha is dual value of the new constraint of next_integer
        subproblem_modified(gen_model,alpha)
        # solve the master problem after adding global_cut_ct
        ms = master_model.solve(log_output=True)
        MM="Master Model Solution and its cost: "
        outF.write(MM)
        outF.write("\n")
        for i in master_model.labels:
            if (master_model.x[i]).solution_value > 0.9:
                ms_label ="master model solution labels: " +  str(master_model.label_table[i])
                outF.write(ms_label)
                outF.write("\n")
        ms_objective = "master model objective: " + str(master_model.objective_value)
        outF.write(ms_objective)
        outF.write("\n")
        # since my global_cut_ct is set individually, i guess there is no need to deal with dual value.
        # I think there is no need to solve the subproblem after adding alpha any more
        # gs = gen_model.solve(log_output = True)
        
    elif m.is_integer() is True:
        pass                            

    toc=time.time()
    print("Time taken for solving is " + str((toc-tic)) + "sec")
    print()
    TimeTaken = "Time taken for solving is "+str((toc-tic)) +"sec"
    outF.write(TimeTaken)
    outF.write("\n")
    
    eecpp_print_solution(master_model, outF)
    outF.close()
    
    
        
def eecpp_solve_default(**kwargs):
    return eecpp_solve(colNumber, rowNumber, label_table,coord_x, coord_y, **kwargs)

def eecpp_print_solution(eecpp_model, outF):
    labels = eecpp_model.labels
    label_table = eecpp_model.label_table
    x_values = {l: eecpp_model.x[l].solution_value for l in labels}
    print("/ Check       / Label    / Label's detail(# of node1, node2,...)         /")
    titleNb = "/ Check       / Label    / Label's detail(# of node1, node2,...)         /"
    outF.write(titleNb)
    outF.write("\n")
    print("/ {} /".format("-" * 70))
    fformat="/{}/".format("-" * 70)
    outF.write(fformat)
    outF.write("\n")
    for l in labels:
        if x_values[l] >= 1e-3:
            label_detail = {a: label_table[a] for a in labels if
                              a==l}
            print(
                "| {:<10g} | {!s:9} | {!s:45} |".format(x_values[l],
                                                       l,
                                                       label_detail))
            print_label_detail = "| {:<10g} | {!s:9} | {!s:45} |".format(x_values[l],
                                                       l,
                                                       label_detail)
            outF.write(print_label_detail)
            outF.write("\n")
    print("| {} |".format("-" * 70))
    fformat = "| {} |".format("-" * 70)
    outF.write(fformat)
    outF.write("\n")

def check_obstacle(obstacles, lastNode, newNode, colNumber, rowNumber):
    sideLength = 1
    flag=1 # initialize flag=1
    for obstacle in obstacles:
        x3 = obstacle % colNumber
        y3 = obstacle // colNumber
        r_xmin = x3 - sideLength/2
        r_xmax = x3 + sideLength/2
        r_ymin = y3 - sideLength/2
        r_ymax = y3 + sideLength/2
        if RSIntersection(r_xmin, r_xmax, r_ymin, r_ymax, lastNode, newNode, colNumber, rowNumber) == 1:
            flag = 0 #flag is variable to replace directly return False of True because we need to test all obstacles
        else:
            flag = 1
        if flag == 0:#once flag change to 0, means one obstacle is hit, thus new node can not be added
            return False
    return True# This means all obstacles are not hit, new node can be considered to be added
"""
 * if [x1, x2] and [x3, x4] (x4 maybe smaller than x3) has interrection or has not，if yes: return 1， if no: return 0
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
def RSIntersection(r_xmin,r_xmax,r_ymin, r_ymax, nodeA, nodeB, colNumber, rowNumber):
    A_x = nodeA % colNumber
    A_y = nodeA // colNumber
    B_x = nodeB % colNumber
    B_y = nodeB // colNumber
    if (A_y == B_y):# line segement is parallel to x axis//线段平行于x轴
        if A_y <= r_ymax and A_y >= r_ymin:
            return IntervalOverlap(r_xmin, r_xmax, A_x,B_x)
        else:
            return 0

	# Echange point A and point B to let point B has bigger y coordinate//AB两点交换，让B点的y坐标最大

    # Exchange node A and node B, let B's y value is bigger
    t = 0
    if A_y > B_y:
       t = A_y
       A_y = B_y
       B_y = t
       t= A_x
       A_x = B_x
       B_x=t
	
    # In line segment//xianduan AB, to find point C and D
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
   
def distance(firstNode, secondNode,coord_x,coord_y):
    j=secondNode
    i=firstNode
    distanceValue=0
    distanceValue=np.hypot(coord_x[i]-coord_x[j],coord_y[i]-coord_y[j])
    return distanceValue

def angle(second_to_lastNode, lastNode, newNode, coord_x, coord_y):
    radians_to_degrees = 180/(math.pi)
    theta_radians=0
    theta_degrees=0
    o=second_to_lastNode
    p=lastNode
    q=newNode
    distance_o_p=distance(o,p,coord_x,coord_y)
    distance_p_q=distance(p,q,coord_x,coord_y)
    distance_o_q=distance(o,q,coord_x,coord_y)
    theta_radians=math.pi-np.arccos(round((distance_o_p**2+distance_p_q**2-distance_o_q**2)/(2*distance_o_p*distance_p_q),2))
    theta_degrees=radians_to_degrees*theta_radians
    return theta_degrees

def global_cut_make_eecpp_master_model(label_table, colNumber, rowNumber, **kwargs):
    label_number = len(label_table)
    nodesNumber= colNumber * rowNumber
    departurePoint = 0
    obstacles = []
    ##
    C=[0] * label_number
    for l in range(label_number):
        C[l] = label_table[l][-1][0]

    # column vector set
    a = [[0 for i in range(nodesNumber)] for j in range(label_number)]
    labels = [l for l in range(label_number)]
    for l in range(label_number):
        for j in range(0, nodesNumber):
            if j in label_table[l]:
                a[l][j] = 1
    #model<->m
    m = Model(name='EECPP_master')
    m.labels = labels
    m.label_table = label_table
    m.label_number = label_number
    m.nodes = [i for i in range(nodesNumber)]
    m.x = m.continuous_var_dict(m.labels,lb=0, ub=1, name="visit") # determine if the label is selected(x=1) or not(x=0)
    # minimize total cost
    m.visiting_cost = m.sum( (C[l] * m.x[l]) for l in labels )
    m.minimize(m.visiting_cost)
    

    m.node_visit_cts=[]
    #
    ##########
    for node in m.nodes:
        if node in obstacles:
            node_visit_ct = m.sum(m.x[l] * a[l][node] for l in labels) >= 0
        else:
            node_visit_ct = m.sum(m.x[l] * a[l][node] for l in labels) >= 1 + 1# obtaining integer solution
        node_visit_ct_name = 'ct_visit{0!s}'.format(node)
        m.node_visit_cts.append(node_visit_ct)
    m.add_constraints(m.node_visit_cts)
    return m 

def global_cut_make_eecpp_generation_model(colNumber, rowNumber, coord_x, coord_y,**kwargs):
    nodesNumber = colNumber * rowNumber
    duals = [0] * nodesNumber
    pi = duals
    departurePoint = 0

    #arc_three_nodes<->arcs
    #c=6.0 battery constraint is 6.0
    
    C = 6.0
    lamda = 0.1164
    garma = 0.0173
    garma = 0 # turning cost is 0
    zero_turn = garma * 180
    C = C + zero_turn
    control_inf = C/lamda+1
    radians_to_degrees = 180/(math.pi)
    distanceValue=0
    theta_radians=0
    theta_degrees=0
    obstacles = []
    Nodes = [i for i in range(nodesNumber) if i not in obstacles and i!=departurePoint]
    # Except departurePoint( arrival point is same as departurePoint) and obstacles
    NodesAndDeparturePoint = Nodes + [departurePoint]
    AllNodes = NodesAndDeparturePoint + obstacles
    edges = [(i,j) for i in NodesAndDeparturePoint for j in NodesAndDeparturePoint]
    arcs = [(i,j,k) for i in NodesAndDeparturePoint for j in NodesAndDeparturePoint for k in NodesAndDeparturePoint]
    distance={(i,j):0 for i,j in edges}
    
    ############
    
    ############
    """
    Need add something to deal with obstacles
    if line segment of i&j interacts with obstacles, c_ij is infinity(math.inf).
    if line segment of i&j does not interact with obstacles, c_ij is european distance.
    """

    c = {(i,j):0 for i,j in edges}
    q = {(i,j,k):0 for i,j,k in arcs}
    distance={(i,j):0 for i,j in edges}
    for i,j in edges:
        distanceValue = np.hypot(coord_x[i]-coord_x[j], coord_y[i]-coord_y[j])
        distance[(i,j)]=distanceValue
        distance_cost = lamda * distanceValue
        c[(i,j)] = distance_cost
    
    for o,p in edges:
        View = check_obstacle(obstacles, o, p, colNumber, rowNumber)
        if View == 0:
            c[(o,p)] = math.inf
        else:
            pass
    turning_cost = 7
    for i,j,k in arcs:
        theta_radians=math.pi-np.arccos(round((distance[i,j]**2+distance[j,k]**2-distance[i,k]**2)/(2*distance[i,j]*distance[j,k]),2))
        theta_degrees=theta_radians*radians_to_degrees
        turning_cost=garma*theta_degrees
        q[(i,j,k)]=turning_cost
        a=math.isnan(turning_cost)
        if a is True:
            turning_cost=0
        else:
            pass
        q[(i,j,k)]=turning_cost
    

    # An arc flow model for the basic EECPP
    gen_model = Model("eecpp_generate_labels")
    gen_model.edges = edges
    gen_model.duals = [0] * nodesNumber
    gen_model.x = gen_model.binary_var_dict(edges, name = 'X') # flow variables, 1 if the agent goes directly from node i to node j
    gen_model.I = gen_model.binary_var_dict(arcs, name = "I") # use I in order to linearly replace x (i,j) x x (j,k)
    d = gen_model.continuous_var_list(AllNodes, name = "D") # d is a dummy variable associated with node i for subtour elimination

    ##### set 4 expr expressions to simplify expressions later
    gen_model.expr_1 = gen_model.sum( c[(i,j)] * gen_model.x[(i,j)] for i,j in edges)
    gen_model.expr_2 = gen_model.sum( gen_model.duals[i] * gen_model.x[(i,j)] for i,j in edges)
    # gen_model.expr_3 has mistake
    gen_model.expr_3 = gen_model.sum((q[(i,j,k)]* gen_model.I[(i,j,k)]) for i,j,k in arcs)
    # turn cost calculation is wrong
    gen_model.expr_4 = gen_model.sum( (q[(i,0,k)] * gen_model.I[(i,0,k)]) for i,k in edges)
    #####
    
    # what is next_integer
    alpha = dual[next_integer]
    gen_model.expr = gen_model.expr_1 - gen_model.expr_2 + gen_model.expr_3 - gen_model.expr_4 - alpha
    gen_model.labelCost = gen_model.expr + gen_model.expr_2
    gen_model.energyCost = gen_model.expr_1 + gen_model.expr_3
    
    gen_model.minimize(gen_model.expr) #(24)
    
    # 2. setup constraint
    gen_model.add_constraint(gen_model.sum( gen_model.x[(0,j)] for j in Nodes) == 1, ctname = None) #(25)
    gen_model.add_constraint(gen_model.x[(0,0)] !=1, ctname= None) # (33)


    #once a drone visits a node, it also departs from the same node. Each node can and can only be visited only once.
    for j in NodesAndDeparturePoint:
        gen_model.add_constraint(gen_model.sum( gen_model.x[(i,j)] for i in NodesAndDeparturePoint)==
                                 gen_model.sum( gen_model.x[(j,k)] for k in NodesAndDeparturePoint), ctname = None) #(31)
        
    for i,j,k in arcs:
        gen_model.add_constraint( gen_model.I[(i,j,k)] >= gen_model.x[(i,j)] + gen_model.x[(j,k)] - 1, ctname = None ) #(26)
        gen_model.add_constraint( gen_model.I[(i,j,k)] <= gen_model.x[(i,j)], ctname = None ) #(27)
        gen_model.add_constraint( gen_model.I[(i,j,k)] <= gen_model.x[(j,k)], ctname = None ) #(28)

    #(29)
#    gen_model.add_constraint( (gen_model.sum(( gen_model.x[(i,j)] * c[(i,j)] ) for i,j in edges) + gen_model.sum(( q[(i,j,k)]* gen_model.I[(i,j,k)] ) for i,j,k in arcs) )<= C, ctname = None)  
    gen_model.add_constraint( gen_model.energyCost <= C, ctname = None )
    #(30) subtour elimination constraint
    for i,j in edges:    
        if j!=departurePoint:
            gen_model.add_indicator( gen_model.x[(i,j)],d[i] + 1 == d[j],name=None)

def add_global_cut(master_model, next_integer):
    labels = master_model.labels
    master_model.global_cut_ct = master_model.sum(master_model.x[l] for l in labels) >= next_integer
    master_model.add_constraint(master_model.global_cut_ct)
    return master_model

def subproblem_modified(gen_model, alpha):
    gen_model.expr = gen_model.expr - alpha
    gen_model.minimize(gen_model.expr)
    return gen_model
    

if __name__ == '__main__':
    s = eecpp_solve_default()