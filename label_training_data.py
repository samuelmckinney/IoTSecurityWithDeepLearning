import pandas as pd 
import os

path = "./data/"
    
filename = os.path.join(path,"benign_traffic.csv")  
benign  = pd.read_csv(filename)

filename2 = os.path.join(path,"benign_traffic_2.csv")  
benign2  = pd.read_csv(filename2)

path = "./data/mirai_attacks"

other_file = os.path.join(path,"scan.csv")  
malicious_scan  = pd.read_csv(other_file)

second_attack = os.path.join(path, "syn.csv")
syn_attack = pd.read_csv(second_attack)

third_attack = os.path.join(path, "ack.csv")
ack_attack = pd.read_csv(third_attack)

fifth_attack = os.path.join(path, "udp.csv")
udp_attack = pd.read_csv(fifth_attack)

sixth_attack = os.path.join(path, "udpplain.csv")
udpplain_attack = pd.read_csv(sixth_attack)

malicious_scan = malicious_scan.assign(label=1)
syn_attack = syn_attack.assign(label=2)
ack_attack = ack_attack.assign(label=3)
udp_attack = udp_attack.assign(label=4)
udpplain_attack = udpplain_attack.assign(label=5)



benign = benign.assign(label=0)
benign2 = benign2.assign(label=0)

frames = [benign2, syn_attack] #, syn_attack, ack_attack, udp_attack, udpplain_attack]

both = pd.concat(frames)



both.to_csv("merged_labeled.csv")




#ecobee thermostat

