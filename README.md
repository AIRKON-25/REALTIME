# REALTIME
실시간에서 동작하는 코드

원격 접속을 한다 cam1: 192.168.0.101, cam3: 192.168.0.106

python -m edge_http --conf 0.01 --class-conf '0:0.05,1:0.5,2:0.5' --weights tensorrt/yolo11m_fp16_real_ces_v4.engine --udp-enable --udp-host "인지_컴퓨터_ip_주소"  --camera-id 1  --homography H/cam1_H.json --roi-path roi/cam1_roi.npz
  
python -m edge_http --conf 0.01 --class-conf '0:0.05,1:0.5,2:0.5' --weights tensorrt/yolo11m_fp16_real_ces_v4.engine --udp-enable --udp-host "인지_컴퓨터_ip_주소"  --camera-id 3  --homography H/cam3_H.json --roi-path roi/cam3_roi.npz
  
==============  
### server
python server.py --tx-host --tx-port --car-count N --log-udp-packets --no-log-pipeline

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

npm run dev -- --host 0.0.0.0




### H 및 roi 생성 방법

## 1. H 생성 방법

1. edge 컴퓨터에서 python utils/camera/camera_output_http.py 실행
2. 본인의 컴퓨터에서 edge_컴퓨터_ip:8080 (ex: 192.168.0.101:8080) 접속해서 이미지 다운로드
3. python utils/make_H/homography_tool.py --cam "방금 다운로드 받은 이미지" --outdir "결과를 저장하고 싶은 폴더" 실행
4. 화살표 등과 같은 특징점들을 실제 이미지와 조감도와 메칭(모든 화살표는 다찍기) 그 다음 엔터 --> 나오는 창에서 투영 잘 되었다 싶으면 s 눌러 저장.
5. 폴더 열어보면 이미지 2개랑 .json 파일이 생기는데, 이게 H 행렬 --> 이걸 edge 컴퓨터로 전송해서 H 행렬로 사용

## 2. roi 생성 방법

1. edge 컴퓨터에서 python utils/camera/camera_output_http.py 실행
2. 본인의 컴퓨터에서 edge_컴퓨터_ip:8080 (ex: 192.168.0.101:8080) 접속해서 이미지 다운로드
3. python utils/roi/create_roi_from_image.py --image "방금 다운로드 받은 이미지" --out "roi/camX_roi.npz" 실행 (X는 카메라 번호)
4. 내가 관심있는 영역(실제로 추론할때 사용할 영역) 설정, 점찍을 수 있고, 점으로 인해서 만들어진 폴리곤 내부가 ROI가 됨 --> 다 하면 s 눌러 저장
5. 저장된 .npz 파일이 roi 파일 --> 이걸 edge 컴퓨터로 전송해서 roi 파일로 사용
