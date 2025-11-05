from net import Net
import subprocess
import time
from wmediumdConnector import w_server
from connect import Interface

Net_1 = Net(name='Net_1', interval=1)
Net_1.add_node(name="11", mac="02:00:00:00:01:00", ip="10.10.10.1/24",position= (140, 0, 0), direction=(-10, 1, 1))
Net_1.add_node(name="12", mac="02:00:00:00:02:00", ip="10.10.10.2/24",position= (-140, 0, 0),direction= (10, 1, 1))
Net_1.add_node(name="13", mac="02:00:00:00:03:00", ip="10.10.10.3/24",position=(-200, 0, 0),direction=(0, 0, 1))
Net_1.add_node("14", "02:00:00:00:00:04", "10.10.10.4/24",(250, 0, 30),(-10, -10, 1))
Net_1.add_node("15", "02:00:00:00:00:05","10.10.10.5/24",(50,50,0),(0, 0, 100))
Net_1.add_node("16", "02:00:00:00:00:06","10.10.10.6/24",(1,20,300),(10,10,10))
Net_1.add_node("17", "02:00:00:00:00:07","10.10.10.7/24",(10,20,30),(1,1,100))


# docker stop 11 &&docker rm 11 && docker stop 12 && docker rm 12&&docker stop 13 && docker rm 13&&
# docker stop 14&&docker rm 14&&docker stop 15 && docker rm 15&&docker stop 16 && docker rm 16&&docker stop 17 && docker rm 17

times={}
Net_1.start_network()
time.sleep(5)
time_all=time.time()
for i in range(80):
    
    time2 = time.time()
    # Net_1.move_nodes()
    print(f"===第{i+1}次测试===")
    Net_1.test_all_links_concurrent()
    time1= time.time()-time2
    times[i]={'第次测试':i+1,'耗时':{time1}}
        

print("===测试全部结束===")
time1=time.time()-time_all
print(f"总共测试时间：{time1}秒") 
time.sleep(5)  
Net_1.plot_all_nodes()

print(times)


# Net_1.select_core_nodes()
Net_1.end_test()

