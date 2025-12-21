# REALTIME
실시간에서 동작하는 코드

원격 접속을 한다 cam1: 192.168.0.101, cam3: 192.168.0.103  
python -m server --conf 0.2 --homography H\cam1_H.json --udp-enable --weights tensorrt\yolo11m_fp16_real_ces_v3.engine --camera-id 1 --udp-host ~  
  
python -m server --conf 0.2 --homography H\cam3_H.json --udp-enable --weights tensorrt\yolo11m_fp16_real_ces_v3.engine --camera-id 3 --udp-host ~  
  
==============  
### server
python server.py --tx-host --tx-port  

### yaw/color 수동 수정  
"cmd": ["flip_yaw", "set_color", "set_yaw", "list_tracks"] 중에 골라쓰면 됨  

- yaw 뒤집고 싶다
{"cmd": "flip_yaw", "track_id": 1}  

- yaw 수정하고 싶다  
{"cmd": "set_yaw", "track_id": 1, "yaw": 180}   

- color 수정하고 싶다
{"cmd": "set_color", "track_id": 1, "color": "red"} 
ㄴ "red", "green", "white", "yellow", "purple" 중에 고르기  

python -c "import socket, json; s=socket.socket(); s.connect(('127.0.0.1',18100)); s.sendall(json.dumps({'cmd':'set_color','track_id':1,'color':'red'}).encode()+b'\n'); s.close()"

### WEB   
node js 설치   

cd autobrain-ui   

npm install   

npm run dev    

