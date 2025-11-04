from wmediumdConnector import WStarter, w_server, WmediumdException,w_gain, SNRLink, ERRPROBLink, w_pos, w_txpower
import time

class Interface:
    def __init__(self,id, name, mac):
        self.i_d = id
        self.name = name
        self.mac = mac
        
    def id(self):
        return self.i_d
    
    def get_station_name(self):
        return self.name
    
    def get_mac(self):
        return self.mac
    
    
def main():
    try:
        # # 创建接口对象
        # sta1 = Interface(id="11",name="sta1", mac="02:00:00:00:01:00")
        # sta2 = Interface(id="22",name="sta2", mac="02:00:00:00:02:00")

        # # 启动 wmediumd 服务
        # print("Starting wmediumd service...")
        # WStarter(
        #     intfrefs=[sta1, sta2],
        #     links=[
        #         ERRPROBLink(
        #             sta1intf=sta1,
        #             sta2intf=sta2,
                    
        #         )
        #     ]
        # )
        
        w_server.connect()
        # # print("wmediumd service started.")

        # # 更新节点的天线增益
        # gain = w_gain(
        #     staintf=sta1,  # 网络接口
        #     sta_gain=5        # 5 dBi 的天线增益
        # )
        # # w_server.update_gain(gain)

        sta3 = Interface(id="33",name="sta3", mac="02:00:00:00:03:00")
        sta4 = Interface(id="44",name="sta4", mac="02:00:00:00:04:00")
        print("Registering interfaces...")
        sta1_id = w_server.register_interface(sta3.get_mac())
        sta2_id = w_server.register_interface(sta4.get_mac())
        print(f"Registered interfaces with IDs: {sta1_id}, {sta2_id}")

        # # 更新链路 SNR
        # print("Updating SNR for the link...")
        # link = SNRLink(
        #     sta1intf=sta3,
        #     sta2intf=sta4,
        #     snr=10
        # )
        # w_server.update_link_snr(link)
        # print("SNR updated successfully.")

        # # 更新链路错误概率
        # print("Updating SNR probability for the link...")
        # errprob_link = SNRLink(
        #     sta1intf=sta1,
        #     sta2intf=sta2,
        #     snr=80
        # )
        # w_server.update_link_snr(errprob_link)
        # print("SNR probability updated successfully.")

        # 更新节点位置
        # print("Updating position for a node...")
        # position = w_pos(
        #     staintf=sta3,
        #     sta_pos=(10.0, 20.0, 30.0)
        # )
        # w_server.update_pos(position)
        # print("Position updated successfully.")

        # # 更新发射功率
        # print("Updating transmission power for a node...")
        
        # txpower = w_txpower(
        #     staintf=sta4,
        #     sta_txpower=20
        # )
        # w_server.update_txpower(txpower)
        # time.sleep(2)
        # print("Transmission power updated successfully.")

    except WmediumdException as e:
        print(f"connect_py Error during wmediumd operations: {str(e)}")
    finally:
        # 断开连接
        print("Disconnecting from wmediumd server...")
        w_server.disconnect()
        print("Disconnected from wmediumd server.")

if __name__ == "__main__":
    main()
    

