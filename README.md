# REALTIME
실시간에서 동작하는 코드

원격 접속을 한다 cam1: 192.168.0.101, cam3: 192.168.0.103  
python -m edge --conf 0.2 --homography H\cam1_H.json --udp-enable --weights tensorrt\yolo11m_fp16_real_ces_v3.engine --camera-id 1 --udp-host ~  
  
python -m edge --conf 0.2 --homography H\cam3_H.json --udp-enable --weights tensorrt\yolo11m_fp16_real_ces_v3.engine --camera-id 3 --udp-host ~  
  
==============  
### server
python server.py --tx-host --tx-port --log-udp-packets --no-log-pipeline    
--car-count N (1-5) locks car external IDs to 1..N  
--log-udp-packets 주면 엣지에서 잘 오는지 확인하는 로그 켜짐  
--no-log-pipeline 주면 트래킹까지 무사히 마치고 제어컴에 뭘 넘긴건지 확인하는 로그 꺼짐  

### yaw/color 수동 수정  
"cmd": ["flip_yaw", "set_color", "set_yaw", "swap_ids", "list_tracks"] 중에 골라쓰면 됨  

- yaw 뒤집고 싶다
{"cmd": "flip_yaw", "track_id": 1}  
python -c "import socket, json; s=socket.create_connection(('127.0.0.1',18100)); s.sendall((json.dumps({'cmd':'flip_yaw','track_id':1,'delta':180})+'\n').encode()); print(s.recv(4096).decode()); s.close()"

- yaw 수정하고 싶다  
{"cmd": "set_yaw", "track_id": 1, "yaw": 180}   
python -c "import socket, json; s=socket.create_connection(('127.0.0.1',18100)); s.sendall((json.dumps({'cmd':'set_yaw','track_id':1,'yaw':90})+'\n').encode()); print(s.recv(4096).decode()); s.close()"

- color 수정하고 싶다
{"cmd": "set_color", "track_id": 1, "color": "red"} 
ㄴ "red", "green", "white", "yellow", "purple" 중에 고르기  
python -c "import socket, json; s=socket.create_connection(('127.0.0.1',18100)); s.sendall((json.dumps({'cmd':'set_color','track_id':1,'color':'red'})+'\n').encode()); print(s.recv(4096).decode()); s.close()"

- 외부 ID끼리 스왑
python -c "import socket, json; s=socket.create_connection(('127.0.0.1',18100)); s.sendall((json.dumps({'cmd': 'swap_ids', 'track_id_a': 1, 'track_id_b': 2})+'\n').encode()); print(s.recv(4096).decode()); s.close()"  

- 트랙 리스트 보고 싶다 
python -c "import socket, json; s=socket.create_connection(('127.0.0.1',18100)); s.sendall((json.dumps({'cmd':'list_tracks'})+'\n').encode()); print(s.recv(4096).decode()); s.close()"  

- car-count가 바뀌었다
python -c "import socket, json; s=socket.create_connection(('127.0.0.1',18100)); s.sendall((json.dumps({'cmd':'set_car_count', 'car_count':4})+'\n').encode()); print(s.recv(4096).decode()); s.close()"  


### WEB   
node js 설치   

cd autobrain-ui   

npm install   

npm run dev    

