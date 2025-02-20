from flask import Flask, render_template
import os
from network_visualization import create_radio_graph

app = Flask(__name__)

@app.route('/')
def index():
    # 确保static目录存在
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # 生成图形
    create_radio_graph()
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 