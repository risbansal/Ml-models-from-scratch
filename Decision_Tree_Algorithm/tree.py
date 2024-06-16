import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import copy
import time
import argparse

start = time.time()



class dtree:
    def __init__(self, data, L = None, R = None, nlist = None, plist = None):
        self.left = L
        self.right = R
        self.value = data
        self.nlist = nlist
        self.plist = plist

def build_tree(Z, E, EV):


    if (len(Z["Y"].unique()) == 1) and (0 in Z["Y"].unique()):
        return dtree(0)
    elif (len(Z["Y"].unique()) == 1) and (1 in Z["Y"].unique()):
        return dtree(1)
    elif len(Z.columns) <= 2:
        col_pos = Z["Y"][Z.iloc[:, 0] == 1]
        pp = col_pos.sum()
        pn = len(col_pos) - pp
        col_neg = Z["Y"][Z.iloc[:, 0] == 0]
        np = col_neg.sum()
        nn = len(col_neg) - np

        if pp > pn:
            r = 1
        else:
            r = 0

        if np > nn:
            l = 1
        else:
            l = 0

        return dtree(Z.columns[0], dtree(l), dtree(r))
    else:
        best, En, Ep, nlist, plist = ig(Z, E, EV)
        Zp = Z[Z[best] == 1]
        Zn = Z[Z[best] == 0]

        if len(Zp.columns) > 2:
            Zp = Zp.drop([best], axis = 1)
        if len(Zn.columns) > 2:
            Zn = Zn.drop([best], axis = 1)


        return dtree(best, build_tree(Zn, En, EV), build_tree(Zp, Ep, EV), nlist, plist)



# Entropy function
def get_entropy(yp, yn):
    if yp == 0 or yn == 0:
        entropy = 0
    else:
        entropy = -(yp / (yp+yn)) * math.log((yp / (yp+yn)), 2) -(yn / (yp+yn)) * math.log((yn / (yp+yn)), 2)

    return entropy

#Variance Function
def get_variance(yp, yn):
    tot = yp + yn
    if yp == 0 or yn == 0:
        var = 0
    else:
        var = (yp * yn) / (tot * tot)
    return var

def ig(Z , E, EV):
    best = None
    Eneg = 0
    Epos = 0
    max_ig = 0

    Z_np = Z.values.T
    for cols in Z_np[:-1]:
        nlist = [0, 0]
        plist = [0, 0]
        for i in range(len(cols)):
            if cols[i] == 0 and Z_np[-1][i] == 0:
                nlist[0] = nlist[0] + 1
            if cols[i] == 0 and Z_np[-1][i] == 1:
                nlist[1] = nlist[1] + 1
            if cols[i] == 1 and Z_np[-1][i] == 0:
                plist[0] = plist[0] + 1
            if cols[i] == 1 and Z_np[-1][i] == 1:
                plist[1] = plist[1] + 1

        if EV == 1:
            Ep = get_entropy(plist[0], plist[1])
            En = get_entropy(nlist[0], nlist[1])

        else:
            Ep = get_variance(plist[0], plist[1])
            En = get_variance(nlist[0], nlist[1])

        tot = nlist[0] + nlist[1] + plist[0] + plist[1]
        Gain = E - ((plist[0] + plist[1]) / tot) * Ep - ((nlist[0] + nlist[1]) / tot) * En


        if max_ig < Gain:
            max_ig = Gain
            best = Z.columns[Z_np.tolist().index(cols.tolist())]
            Eneg = En
            Epos = Ep
    return (best, Eneg, Epos, nlist, plist)





def display_tree(tree):
    root = tree
    print(root.value)
    print(root.value, " (L) ", root.left.value)

    print(root.value, " (R) ", root.right.value)

    val_l = root.left
    val_r = root.right

    print("Left Subtree")
    while val_l.value != 0 and val_l.value != 1:
        print(val_l.value, " (L) ", val_l.left.value)
        print(val_l.value, " (R) ", val_l.right.value)
        val_l = val_l.left

    print("Right Subtree")
    while val_r.value != 0 and val_r.value != 1:
        print(val_r.value, " (L) ", val_r.left.value)
        print(val_r.value, " (R) ", val_r.right.value)
        val_r = val_r.right


def add_to_list(tree, level):
    if tree is None:
        return
    if level == 1:
        node_list.append(tree)
    elif level > 1:
        add_to_list(tree.left, level - 1)
        add_to_list(tree.right, level - 1)

def tree_depth(node):
   if node is None:
       return 0
   else:
       lh = tree_depth(node.left)
       rh = tree_depth(node.right)

       if lh > rh:
           return lh + 1
       else:
           return rh + 1

def bfs(tree):
    bfs_list = []
    depth = tree_depth(tree)
    for i in range(1, depth + 1):
        add_to_list(tree, i)
        temp = node_list.copy()
        bfs_list.append(temp)
        node_list.clear()
    return bfs_list


def tester(tree, td):
    tester_list = []
    td1 = td.iloc[:, 0:-1].values
    for r in td1.tolist():
        tester_list.append(tree_traversal(tree, r))
    tester_p = pd.Series({"PY" : tester_list})
    c = (td.Y == tester_p.PY).sum()

    acurracy = c / len(tester_p.PY)
    return acurracy, tester_p


def tree_traversal(tree, r):
    if tree.value == 0:
        return 0
    elif tree.value == 1:
        return 1
    else:
        i = tree.value.replace("X", "")
        v = r[int(i)]
        if v == 0:
           return tree_traversal(tree.left, r)
        else:
            return tree_traversal(tree.right, r)

node_list = []

def prune_d(bfs_list, max_d, tree, vd):

    copy_list = []
    if max_d < len(bfs_list):
        for n in bfs_list[max_d]:
            if (n.value != 0) and (n.value != 1):
                copy_tree = copy.deepcopy(n)
                ind = bfs_list[max_d].index(n)
                copy_list.append((copy_tree,ind))
                if (n.nlist[1] + n.plist[1]) > (n.nlist[0] + n.plist[0]):
                    n.value = 1
                else:
                    n.value = 0
                n.left = None
                n.right = None

        for c, i in copy_list:
            bfs_list[max_d][i].value = c.value
            bfs_list[max_d][i].left = c.left
            bfs_list[max_d][i].right = c.right

        accuracy3, dprune_test = tester(tree, vd)
        return accuracy3


def prune(bfs_list, max_d, tree, td):
    if max_d <= len(bfs_list):
        for n in bfs_list[max_d-1]:
            if (n.value != 0) and (n.value != 1):
                if (n.nlist[1] + n.plist[1]) > (n.nlist[0] + n.plist[0]):
                    n.value = 1
                else:
                    n.value = 0
                n.left = None
                n.right = None

        accuracy3, dprune_test = tester(tree, td)
        return accuracy3



def prune_r(bfs_list, accuracy, tree, vd):
    accuracy1 = accuracy
    bfs_list.reverse()
    for l in bfs_list[:-1]:
        for n in l:
            if n.value != 0 and n.value != 1:
                t_node = n.value
                t_left = n.left
                t_right = n.right
                neg = n.nlist[0] + n.plist[0]
                pos = n.nlist[1] + n.plist[1]
                if pos < neg:
                    n.value = 0
                else:
                    n.value = 1
                n.left = None
                n.right = None
                accuracy2, rprune_test = tester(tree, vd)
                if accuracy2 < accuracy1:
                    n.value = t_node
                    n.left = t_left
                    n.right = t_right
                else:
                    accuracy1 = accuracy2
    return accuracy1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-EV', '--ev', type=int, default = 1)
    parser.add_argument('-p_type', '--pruning', type=int, default = 0)
    parser.add_argument('-train_data', '--trdata', type=str)
    parser.add_argument('-test_data', '--tdata', type=str)
    parser.add_argument('-valid_data', '--vdata', type=str)



    arg = parser.parse_args()

    EV = arg.ev
    prune_type = arg.pruning
    train_file = arg.trdata
    valid_file = arg.vdata
    test_file = arg.tdata

    # EV = 0
    # prune_type = 1
    #
    #
    # # Reading data
    # train_file = "data/train_c500_d5000.csv"
    # test_file = "data/test_c500_d5000.csv"
    # valid_file = "data/valid_c500_d5000.csv"


    trd = pd.read_csv(train_file, header=None)
    vd = pd.read_csv(test_file, header=None)
    td = pd.read_csv(valid_file, header=None)

    n_columns = len(trd.columns)
    n_rows = len(trd.index)
    trd.columns = range(n_columns)
    # X = trd.iloc[:, 0:n_columns]

    td.columns = ["X{}".format(i) for i in range(n_columns)]
    test_list = list(trd.columns)
    test_list[-1] = "Y"
    td.columns = test_list

    vd.columns = ["X{}".format(i) for i in range(n_columns)]
    validate_list = list(trd.columns)
    validate_list[-1] = "Y"
    vd.columns = validate_list

    trd.columns = ["X{}".format(i) for i in range(n_columns)]
    train_list = list(trd.columns)
    train_list[-1] = "Y"
    trd.columns = train_list

    yp = 0
    yn = 0
    Y = trd.iloc[:,-1]

    for i in Y:
        yp = yp + i
        yn = n_rows - yp



    Ey = get_entropy(yp,yn)

    Vy = get_variance(yp,yn)

    if EV == 1:
        tree = build_tree(trd, Ey, EV)
    else:
        tree = build_tree(trd, Vy, EV)

    accuracy, test_labels = tester(tree,td)

    depth = tree_depth(tree)
    print("Accuracy without Pruning:", accuracy)
    print("Depth of tree:", depth)


    bfs_list = bfs(tree)

    if prune_type == 1:
        accuracy_r = prune_r(bfs_list, accuracy, tree, vd)
        print("Accuracy for reduced error pruning on Valid set:", accuracy_r)
        accuracy_rt, list_t = tester(tree, td)
        print("Accuracy for reduced error pruning on Test set:", accuracy_rt)

    if prune_type == 2:
        max_depth = [100, 50, 20, 15, 10, 5]
        max_ac = 0.0
        for md in max_depth:
            accuracy_d = prune_d(bfs_list, md, tree, vd)
            if accuracy_d == None:
                accuracy_d = 0

            elif max_ac < accuracy_d:
                max_ac = accuracy_d
                max_d = md



        print("Max Accuracy for depth based pruning on Valid set:", max_ac)
        print("Max depth for pruning:", max_d)
        test_accuracy = prune(bfs_list, max_d, tree, td)
        print("Test accuracy after Pruning:", test_accuracy)


    #Random Forest
    rf = RandomForestClassifier(n_estimators = 80, random_state = 431)
    rf.fit(trd.iloc[:, 0:-1].values, trd["Y"].values)
    rf_p = rf.predict(td.iloc[:, 0:-1].values)
    rf_ac = metrics.accuracy_score(td["Y"].values, rf_p)

    print("Random Forest Accuracy", rf_ac)

main()



#display_tree(tree)
etime = time.time() - start
print("Execution time:", etime)


