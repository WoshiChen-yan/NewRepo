

class wlsintf(object):
    def __init__(self,name,node,mac,ip,port=None,link=None,prefixlen=None,txpower=30,type='ibss',freq=2432):
        self.node=node
        self.name=name
        self.port=port
        self.link=link
        self.mac=mac
        self.type=type
        self.freq=freq
        self.txpower=txpower
        self.ip,self.prefixLen=ip,prefixlen
        
        # print(self.mac)
        # print(self.ip)
        self.down()
        if self.type == 'mp':
            self.iw_set_type_mp()
                     #设置无线网卡为mesh pooint模式
            self.setMAC()
            self.setIP()
            self.iw_set_txpower()
            self.up()
            self.iw_set_channels()
            self.iw_set_mesh_join(1,self.freq)
        if self.type == 'ibss':
            self.iw_set_type_ibss()
                    ##设置无线网卡为ibss （adhoc）模式
            self.setMAC()
            self.setIP()
            self.iw_set_txpower()
            self.up()
            self.iw_set_channels()
            self.iw_set_ibss_join(1,self.freq)
         
        # print(self.cmd(f"ip route get 10.10.10.1"))
        # 可能后续使用 batman-adv协议进行mesh组网 用于
        # 自动获取拓扑与前置的路由
         

    def cmd(self,*args):
        return self.node.cmd(*args)
    
    def down(self):
        return self.node.cmd('ip link set wlan{} down'.format(self.name))
    
    def up(self):
        return self.node.cmd('ip link set wlan{} up'.format(self.name))
        
    def iw(self,*args):
        return self.cmd('iw dev wlan{}'.format(self.name),*args)

    def iw_set_type_ibss(self):
        return self.iw('set type ibss')
    
    def iw_set_type_mp(self):
        return self.iw('set type mp')

    def iw_set_bitrates(self,*args):
        return self.iw('set bitrates ',*args)

    def iw_set_channels(self):
        return self.iw('set channel 149')

    def iw_set_ibss_join(self,ssid,freq):
        return self.iw('ibss join {} {}'.format(ssid,freq))
    
    def iw_set_mesh_join(self,ssid,freq):
        return self.iw('mesh join ',ssid,'freq',freq)

    def iw_scan(self):
        return self.iw('scan'.format(self.name))
    
    def iw_link(self):
        return self.iw('link'.format(self.name))
    
    def iw_station_dump(self):
        return self.iw('station dump')
    
    def iw_survey_dump(self):
        return self.iw('survey dump')
    
    def iw_set_txpower(self):
        return self.iw('set txpower {}dBm'.format(self.txpower))
    
    def iw_get_rssi(self):
        return self.iw("link")
    
    def iw_info(self):
        return self.iw('info')
    
    def setMAC(self):
        if not self.mac:
            print("缺少指定的mac地址")
            return
        comm=self.cmd('ip link set dev wlan{} address {}'.format(self.name, self.mac))
        print(comm)
        print("节点 {} 的 wlan{} 端口的mac地址：{}已配置".format(self.node.name, self.name, self.mac))

    def setIP(self):
        if not self.ip:
            print("缺少指定的ip地址")
            return
        if '/' in self.ip:
            ipstr=self.ip
            self.ip, self.prefixLen = ipstr.split('/')
        else:
            if self.prefixLen is None:
                raise Exception(f'No prefix length set for IP address {self.ip}')
                                
        self.cmd('ip address add {}/{} dev wlan{}'.format(self.ip, self.prefixLen, self.name))
        print("节点 {} 的 wlan{} 端口的ip地址：{}/{}已配置".format(self.node.name, self.name, self.ip, self.prefixLen))