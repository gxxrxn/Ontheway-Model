from flask import Flask, render_template, request, redirect, make_response
import recommend
import json
import numpy

app = Flask(__name__)
@app.route('/', methods=['POST', 'GET']) # 1번째 페이지
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST', 'GET']) # 3번째 페이지
def map():
    if request.method == 'POST':
        res = request.get_json()
        print('😜',res)
        # res = {'0': {'name': '부산올림픽공원', 'lat': 35.1646992, 'lon': 129.1309505}, '1': {'name': '부산 박물관', 'lat': 35.1295178, 'lon': 129.091957}, '2': {'name': '부산대학교 대학로', 'lat': 35.1513153, 'lon': 129.033806}, '3': {'name': '부산영화체험박물관', 'lat': 35.101702, 'lon': 129.0315715}, '4': {'name': '부산관광안내소', 'lat': 35.2853903, 'lon': 129.0934332}}
        rec = recommend.result(res['place'], res['pos'])
        print(rec)

        result = json.dumps(rec, ensure_ascii=False)
        return make_response(result)
        
if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
