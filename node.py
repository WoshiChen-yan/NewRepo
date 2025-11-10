import docker
import sys
import os
import re
import numpy as np
import pty
import time
import subprocess
from module import Mac80211Hwsim
from Wirelesslink import wlsintf
from wmediumdConnector import WStarter, w_server, WmediumdException,w_gain, SNRLink, ERRPROBLink, w_pos, w_txpower

nodes=[]

# 找到 self.nodes 列表的辅助函数
def get_node_by_mac(mac):
    for node in nodes:
        if node.mac == mac:
            return node
    return None

class Node(object):
    portBase = 0
    def __init__(self, id,name=None, mac=None , is_core=False,
                 ip=None,position=None,direction=None,
                 txpower=30,type='ibss',net_name=None):
        self.name = name if name else 'wlan' + str(id)
        self.intf_id = id
        self.intfs = {}
        self.ports = {}
        self.nameToIntf = {}  
        self.mac = mac
        self.ip = ip
        self.wintf=None
        self.position=position
        self.direction=direction
        self.type=type
        self.neighbors={}
        self.link_quality_history = []  # 用于存储链路质量历史记录
        self.net_name=net_name
        self.txpower=txpower###对于绝大多数地面基站来说，使用20的发射功率足够  ，相当于100mv的功率
        docker.from_env().containers.run('ubuntu', 
                                         detach=True, 
                                         tty=True, 
                                         network=None,
                                         privileged=True,
                                         labels={'type': self.type},
                                         name=self.name)  # 创建容器实例
        self.create_netns()
        print("节点{} ID为{}已创建".format(self.name , self.intf_id))
        
        
        

    def get_instance(self):
        return docker.from_env().containers.get(self.name)

    # 创建相应的namespace
    def create_netns(self):
        node_name = self.name
        if not (os.path.exists("/var/run/netns/" + node_name)):
            while True:
                try:
                    container = self.get_instance()
                    if container.attrs['State']['Running']:
                        nodes.append(self)
                        break
                    else:
                        time.sleep(1)

                except:
                    print("等待节点 {}的namespace创建 ...".format(node_name))
                    sys.stdout.flush()
                    time.sleep(1)
                    pass

            ##将链接容器网络命名空间与外部主机的连接在一起方便后续 网卡驱动 的放入
            pid = container.attrs['State']['Pid']
            print("{} has pid={}".format(container, pid))
            if os.path.islink("/var/run/netns/" + node_name):
                os.unlink("/var/run/netns/" + node_name)
            os.symlink("/proc/{}/ns/net".format(pid),
                       "/var/run/netns/" + node_name)


    def set_position(self, position):
        self.position = position
        w_position=w_pos(self.position[0],self.position[1])
    def newPort(self):
        if len(self.ports) > 0:
            return max(self.ports.values()) + 1
        return self.portBase

    def addIntf(self, intf, port=None):
        if port is None:
            port = self.newPort()
        self.intfs[port] = intf
        #self.ports[intf] = port
        self.nameToIntf[intf.name] = intf

    def delIntf(self, intf):
        port = self.ports.get(intf)
        if port is not None:
            del self.intfs[port]
            #del self.ports[intf]
            del self.nameToIntf[intf.name]

    def defaultIntf(self):
        ports = self.intfs.keys()
        if ports:
            return self.intfs[min(ports)]


    def deleteIntfs(self):
        for intf in list(self.intfs.values()):
            intf.delete()

    def remove_node(self):
        try:
            instance = self.get_instance()
            # 删除容器
            instance.remove(force=True)

            print("节点{}已成功删除".format(self.name))
        except docker.errors.NotFound:
            print("找不到节点{}".format(self.name))
        except Exception as e:
            print("删除节点{}时出现错误:{}".format(self.name, e))

    def setMAC(self, mac, intf=None):
        return self.wintf.setMAC(mac)
    
    def id(self):
        return self.intf_id
            
    def setIP(self, ip, prefixLen=8, intf=None, **kwargs):
        return self.wintf.setIP(ip, prefixLen, **kwargs)

    def IP(self, intf=None):
        return self.wintf.ip

    def get_mac(self, intf=None):
        return self.wintf.mac
    
    def get_name(self, intf=None):
        return self.wintf.name
    

    def setParam(self, results, method, **param):
        name, value = list(param.items())[0]
        if value is None:
            return None
        f = getattr(self, method, None)
        if not f:
            return None
        if isinstance(value, list):
            result = f(*value)
        elif isinstance(value, dict):
            result = f(**value)
        else:
            result = f(value)
        results[name] = result
        return result
    
    def cmd(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list):
            cmd = args[0]
            # Allow sendCmd( cmd, arg1, arg2... )
        elif len(args) > 0:
            cmd = args
        else:
            cmd = args
            # Convert to string
        if not isinstance(cmd, str):
            cmd = ' '.join([str(c) for c in cmd])
        instance = self.get_instance()
        exec_id = instance.exec_run(cmd, tty=False)

        output = exec_id.output.decode('utf-8').strip()
        return output

    def __str__(self):
        return self.name

    def addwlans(self):
        #Mac80211Hwsim(self,on_the_fly=True)
        # self.set_ovsbridge()
        wintf=wlsintf(name=self.name,node=self,mac=self.mac,ip=self.ip)
        # time.sleep(3)
        # print(wintf.iw_info())
        # print(wintf.iw_scan())
        # print(wintf.iw_station_dump())
        # print(wintf.iw_survey_dump())
        # print(wintf.iw_get_rssi())
        self.wintf=wintf
        
        
    def calculate_siganal_impact(self,target_node):
        
        doppler_shift = self.calculate_doppler_shift(target_node)
        carrier_freq=2.4e9
        
        freq_offset_impact=abs(doppler_shift/carrier_freq)
        # 计算距离影响
        if freq_offset_impact>0.01:#超过 频率偏移量0.1%才会影响
            signal_degretion=1-(freq_offset_impact * 100)
        else:
            signal_degretion=1
            
        return doppler_shift,signal_degretion
    
             
    def calculate_relative_velocity(self,node2):
        
        pos1=np.array(self.position)
        pos2=np.array(node2.position)
        vel1=np.array(self.direction)
        vel2=np.array(node2.direction) 
        
        speed1=np.linalg.norm(vel1)
        speed2=np.linalg.norm(vel2)
        if speed1==0 or speed2==0:
            return 0
        direction=pos2-pos1
        norm=np.linalg.norm(direction)
        if norm==0:
            return 0
        direction=direction/norm
        v_rel=np.dot(vel1-vel2,direction)
        return v_rel
    
    # 计算多普勒频移
    def calculate_doppler_shift(self,target_node,carrier_freq=2.4e9):
        v_rel= self.calculate_relative_velocity(target_node)
        if v_rel==0:
            return 0
        c=3e8
        doppler_shift=(carrier_freq/c)*v_rel
        return doppler_shift #Hz
    
 
    def get_distance(self,target_node):
        position_1=np.array(self.position)
        position_2=np.array(target_node.position)
        distance=np.linalg.norm(position_1-position_2)
        return distance
    
    def get_angle(self,target_node):
        # 计算方向夹角
        position_1=np.array(self.position)
        position_2=np.array(target_node.position)
        vec_to_target = position_2 - position_1  # 使用正确的方向向量
        direction_vector= np.array(self.direction)  # 源节点的方向向量
        vec_to_target = vec_to_target / (np.linalg.norm(vec_to_target) )  # 归一化
        direction_vector_norm= direction_vector / (np.linalg.norm(self.direction) )  # 归一化源节点方向
        cos_theta = np.dot(vec_to_target, direction_vector_norm)
        angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi 
        return angle
        
    
    def get_rssi(self, macs):
        """
        (MODIFIED)
        运行 iw dev wlanX station dump 并返回一个包含 RSSI 和 Bitrate 的字典
        格式: { 'mac': {'rssi': -50.0, 'bitrate': 144.4}, ... }
        不再修改 self.link_quality_history
        """
        if not macs:
            return {}
        
        # 将 macs 转换为小写 set 以便快速查找
        target_macs = set(m.lower() for m in macs)
            
        command = f"iw dev {self.name} station dump"
        iw_output = self.cmd(command) # 这将使用 docker exec_run
        
        results = {}
        current_mac = None
        
        for line in iw_output.splitlines():
            if line.strip().startswith("Station"):
                try:
                    mac = line.split()[1].lower()
                    if mac in target_macs:
                        current_mac = mac
                        # 为这个 MAC 初始化一个空字典
                        if current_mac not in results:
                            results[current_mac] = {}
                    else:
                        current_mac = None # 不是我们关心的 MAC
                except IndexError:
                    current_mac = None
                continue
            
            if current_mac:
                # 寻找 'signal' (平均信号强度)
                if "signal" in line:
                    try:
                        rssi = float(line.split()[2])
                        results[current_mac]['rssi'] = rssi
                    except (ValueError, IndexError):
                        pass # 解析 RSSI 失败
                
                # 寻找 'tx bitrate' (发送速率)
                if "tx bitrate" in line:
                    try:
                        bitrate = float(line.split()[2])
                        results[current_mac]['bitrate'] = bitrate
                    except (ValueError, IndexError):
                        pass # 解析 bitrate 失败
                
                # 如果没有 'tx bitrate'，回退到 'rx bitrate'
                elif "rx bitrate" in line and 'bitrate' not in results[current_mac]:
                    try:
                        bitrate = float(line.split()[2])
                        results[current_mac]['bitrate'] = bitrate
                    except (ValueError, IndexError):
                        pass # 解析 bitrate 失败

        return results
        
    def get_latency(self, ips):
        """
        (MODIFIED)
        运行 fping 并返回一个包含延迟和丢包的字典
        格式: { 'ip': {'latency': 10.0, 'loss': 0.0}, ... }
        不再修改 self.link_quality_history
        """
        if not ips:
            return {}
        
        ip_list_str = ' '.join(ips)
        # 关键: 我们需要 fping 的摘要 (在 stderr 上)
        # 我们使用 2>&1 将 stderr 重定向到 stdout，以便 self.cmd() 可以捕获它
        command = f"fping -c 3 -q -t 100 {ip_list_str} 2>&1"
        
        fping_output = self.cmd(command)
        results = {}

        # fping -q 的摘要输出格式为:
        # 10.10.10.1 : xmt/rcv/%loss = 3/3/0%, min/avg/max = 1.10/1.23/1.39
        # 10.10.10.3 : xmt/rcv/%loss = 3/0/100%
        
        for line in fping_output.splitlines():
            if "loss" not in line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            ip = parts[0]
            loss_percent = -1.0
            avg_latency = 9999.0

            try:
                # 解析 "xmt/rcv/%loss = 3/3/0%"
                loss_str = parts[4].strip('%,')
                loss_percent = float(loss_str)
            except (ValueError, IndexError):
                continue # 解析 loss 失败

            # 如果可达 (loss < 100%)，则解析延迟
            if loss_percent < 100.0 and len(parts) > 7:
                try:
                    # 解析 "min/avg/max = 1.10/1.23/1.39"
                    avg_latency = float(parts[7].split('/')[1])
                except (ValueError, IndexError):
                    # 无法解析 avg_latency，保持 9999.0
                    pass
            
            results[ip] = {'latency': avg_latency, 'loss': loss_percent}

        return results
   
        
    # <--- MODIFIED: 新增函数，用于构建完整的状态向量 ---_>
    def get_link_quality_by_mac(self, target_mac):
        """
        获取到特定MAC的完整链路质量，用于状态向量。
        """
        target_node = get_node_by_mac(target_mac)
        if not target_node:
            return None

        # 1. 从历史记录中查找最新的 延迟/丢包/RSSI
        latest_record = None
        for record in reversed(self.link_quality_history):
            if record.get('mac') == target_mac:
                latest_record = record
                break
        
        # 2. 实时计算物理信息
        distance = self.get_distance(target_node)
        doppler_shift, signal_impact = self.calculate_siganal_impact(target_node)
        
        if latest_record:
            return {
                'distance': distance,
                'latency': latest_record.get('latency', 9999.0),
                'loss': latest_record.get('loss', 100.0),
                'rssi': latest_record.get('rssi', -100.0),
                'bitrate': latest_record.get('bitrate', 0.0),
                'doppler_shift': doppler_shift
            }
        else:
            # 如果历史记录中没有 (例如，第一次运行)
            return {
                'distance': distance,
                'latency': 9999.0,
                'loss': 100.0,
                'rssi': -100.0,
                'bitrate': 0.0,
                'doppler_shift': doppler_shift
            }

    
    def get_neighbor(self):
        # 获取邻居信息
        cmd1=self.cmd(f"iw dev wlan{self.name} station dump")
        # print(cmd1)
        neighbors = []
        for line in cmd1.split('\n'):
            if 'Station' in line:
                mac = line.split()[1]
                neighbors.append(mac)
        print(f"节点{self.name}的邻居节点: {neighbors}")
       
    # def set_ovsbridge(self,port=6633):
    #     self.cmd(f"service openvswitch-switch start")
    #     bridge=f"br_{self.name}"
    #     self.cmd(f"ovs-vsctl add-br {bridge}")
    #     self.cmd(f"ovs-vsctl add-port {bridge} wlan{self.name}")
    #     cmd_1=self.cmd(f"ovs-vsctl set-controller {bridge} tcp:192.168.11.129:{port}")
    #     print(cmd_1)
    #     cmd_2=self.cmd(f"ovs-vsctl show")
    #     print(cmd_2)
    


