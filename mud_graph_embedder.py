import pandas as pd 
import sys
import os


#Working Graph Representation of MUD profiles. Provide arugments of mud csv, device mac, and gateway mac

class graph():
    def __init__(self, node_num = 0, label = None, name = None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.features = []
        self.succs = []
        self.preds = []
        self.edges = []
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
                
    def add_node(self, feature = []):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])
        
    def add_edge(self, u, v, srcEth, destEth, local, ethType, proto, srcIp, dstIp, srcPort, dstPort): #u and v are nodes - one should always be the device
        self.succs[u].append(u)
        self.preds[v].append(v)

        self.edges.append([u, v, srcEth, destEth, local, ethType, proto, srcIp, dstIp, srcPort, dstPort])

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret


if len(sys.argv) != 4:
    print("usage: <mud.csv> <deviceMac> <gatewayMac>")

path = "./data/"
filename2  = os.path.join(path, sys.argv[1])
mud = pd.read_csv(filename2)

mud = mud.drop(['priority'], axis=1)

#mud = mud.drop(['icmpType', 'icmpCode'], axis=1) #why doesn't this work?

mud_graph = graph( label = 0, name='netatmoweatherstation_mudgraph')


print(mud)
mac_str = sys.argv[2] 
gateway_str = sys.argv[3]



mud['srcMac'][mud['srcMac'] == '<deviceMac>'] = mac_str
mud['dstMac'][mud['dstMac'] == '<deviceMac>'] = mac_str
mud['dstMac'][mud['dstMac'] == '<gatewayMac>'] = gateway_str
mud['srcMac'][mud['srcMac'] == '<gatewayMac>'] = gateway_str

mud_graph.add_node([mac_str])

for item in mud.itertuples():
    if item.srcMac == mac_str: #the traffic rule is from the device
        if item.dstIp == '*':
            if [item.dstMac] not in mud_graph.features:
                mud_graph.add_node([item.dstMac])
            mud_graph.add_edge(mud_graph.node_num-1, mud_graph.features.index([item.dstMac]), mud_graph.features[0], item.dstMac, True, item.ethType, item.ipProto, item.srcIp, item.dstIp, item.srcPort, item.dstPort)
        else:
            if [item.dstIp] not in mud_graph.features:
                mud_graph.add_node([item.dstIp])
            mud_graph.add_edge(mud_graph.node_num-1, mud_graph.features.index([item.dstIp]), mud_graph.features[0], item.dstMac, False, item.ethType, item.ipProto, item.srcIp, item.dstIp, item.srcPort, item.dstPort)

    if item.dstMac == mac_str:
        if [item.srcMac] not in mud_graph.features:
            mud_graph.add_node(item.srcMac)
        mud_graph.add_edge(mud_graph.features.index([item.dstMac]), mud_graph.node_num-1, item.srcMac, mud_graph.features[0], False, item.ethType, item.ipProto, item.srcIp, item.dstIp, item.srcPort, item.dstPort)
    
print("graph size is: ")
print(mud_graph.node_num)


